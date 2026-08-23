// openinfer-core is the Go backend of OpenInfer Studio. The desktop launcher
// starts it with a random session token, a loopback port and the launcher
// PID; the backend prints a structured readiness line on stdout and exits
// when the parent desktop process disappears.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/openinfer/openinfer-studio/internal/api"
	"github.com/openinfer/openinfer-studio/internal/auth"
	"github.com/openinfer/openinfer-studio/internal/chat"
	"github.com/openinfer/openinfer-studio/internal/config"
	"github.com/openinfer/openinfer-studio/internal/database"
	"github.com/openinfer/openinfer-studio/internal/diagnostics"
	"github.com/openinfer/openinfer-studio/internal/downloads"
	"github.com/openinfer/openinfer-studio/internal/hardware"
	"github.com/openinfer/openinfer-studio/internal/hostit"
	"github.com/openinfer/openinfer-studio/internal/huggingface"
	"github.com/openinfer/openinfer-studio/internal/instances"
	"github.com/openinfer/openinfer-studio/internal/models"
	"github.com/openinfer/openinfer-studio/internal/proxy"
	"github.com/openinfer/openinfer-studio/internal/quantize"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
	"github.com/openinfer/openinfer-studio/internal/version"
	"github.com/openinfer/openinfer-studio/migrations"
)

func main() {
	var (
		tokenFlag = flag.String("token", "", "session token issued by the desktop launcher (required)")
		portFlag  = flag.Int("port", 0, "loopback port for the control API (0 = choose)")
		ppidFlag  = flag.Int("ppid", 0, "parent desktop PID; backend exits when it disappears")
		dataDir   = flag.String("data-dir", "", "override application data directory (tests/portable mode)")
		selfTest  = flag.Bool("selftest", false, "start, print readiness, and exit (CI smoke test)")
		showVer   = flag.Bool("version", false, "print version and exit")
	)
	flag.Parse()

	if *showVer {
		fmt.Println(version.Summary())
		return
	}

	if *tokenFlag == "" {
		fmt.Fprintln(os.Stderr, `{"ready":false,"error":"missing --token"}`)
		os.Exit(2)
	}

	layout, err := config.Open(*dataDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, `{"ready":false,"error":%q}`+"\n", "creating app directories: "+err.Error())
		os.Exit(1)
	}

	logs := diagnostics.NewManager(layout.AppLogs)
	defer logs.Close()
	log := logs.Logger("core", slog.LevelInfo)
	log.Info("openinfer-core starting",
		"version", version.Version(), "commit", version.Commit,
		"goos", runtime.GOOS, "goarch", runtime.GOARCH, "data", layout.DataDir)

	db, err := database.Open(layout.Database, migrations.FS)
	if err != nil {
		log.Error("database open failed", "err", err)
		fmt.Fprintf(os.Stderr, `{"ready":false,"error":%q}`+"\n", "database: "+err.Error())
		os.Exit(1)
	}
	defer db.Close()
	settings := config.NewSettings(db.DB)

	// Services.
	hub := api.NewHub()
	hf := huggingface.NewClient()
	if t := auth.LoadHuggingFaceToken(); t != "" {
		hf.SetToken(t)
	}
	dl := downloads.NewManager(db.DB, layout.Partial, hub, logs.Logger("downloads", slog.LevelInfo).Logger,
		func(dir string) uint64 { return hardware.Detect(dir, dir).DiskFreeModels })
	dl.SetAuthHeader(func() string {
		if t := auth.LoadHuggingFaceToken(); t != "" {
			return "Bearer " + t
		}
		return ""
	})
	if n := settings.Get("downloads.concurrency", ""); n != "" {
		var v int
		if _, err := fmt.Sscanf(n, "%d", &v); err == nil {
			dl.SetConcurrency(v)
		}
	}
	if n := settings.Get("downloads.connections", ""); n != "" {
		var v int
		if _, err := fmt.Sscanf(n, "%d", &v); err == nil {
			dl.SetConnections(v)
		}
	}
	if err := dl.RecoverAfterRestart(); err != nil {
		log.Warn("download recovery failed", "err", err)
	}

	lib := models.NewLibrary(db.DB, layout.Models, hub, logs.Logger("models", slog.LevelInfo).Logger)
	// Refresh library metadata when the parser schema bumped or on-disk
	// GGUF files look newer than the DB. Runs in the background so readiness
	// is not blocked; the UI reloads on library.scanned.
	go func() {
		stored := settings.Get("library.metadata_schema", "")
		did, n, reason, err := lib.EnsureFresh(stored)
		if err != nil {
			log.Warn("startup library refresh failed", "err", err)
			return
		}
		if !did {
			log.Info("library metadata fresh", "schema", models.MetadataSchemaVersion)
			return
		}
		if err := settings.Set("library.metadata_schema", models.MetadataSchemaVersion); err != nil {
			log.Warn("saving library metadata schema failed", "err", err)
		}
		log.Info("library refreshed on startup", "models", n, "reason", reason)
	}()
	rt := runtimes.NewManager(db.DB, layout.Runtimes, dl, hub, logs.Logger("runtimes", slog.LevelInfo).Logger)
	im := instances.NewManager(db.DB, rt, lib, hub, logs.Logger("instances", slog.LevelInfo).Logger,
		layout.InstLogs, layout.Temp, layout.CacheDir)
	if n := settings.Get("instances.max_loaded", ""); n != "" {
		var v int
		if _, err := fmt.Sscanf(n, "%d", &v); err == nil {
			im.SetMaxLoaded(v)
		}
	}

	qm := quantize.NewManager(db.DB, layout, rt, lib, hub, logs.Logger("quantize", slog.LevelInfo).Logger,
		func(dir string) uint64 { return hardware.Detect(dir, dir).DiskFreeModels },
		func() *hardware.Info { return hardware.Detect(layout.Models, layout.Runtimes) })
	qm.SetConvertDeps(hf, dl)
	qm.SetInstanceHooks(func() []string { return api.LoadedModelIDs(im) }, im)
	if err := qm.RecoverAfterRestart(); err != nil {
		log.Warn("quant job recovery failed", "err", err)
	}

	// Forward structured log records to WS clients for the live Logs page.
	go func() {
		ch, unsub := logs.Subscribe()
		defer unsub()
		for e := range ch {
			hub.Publish("log.entry", e)
		}
	}()

	// Rescan library when a model download completes.
	go func() {
		ch, unsub := hub.Subscribe()
		defer unsub()
		for env := range ch {
			if env.Event != "download.state_changed" {
				continue
			}
			payload, _ := env.Payload.(map[string]any)
			if payload == nil || payload["state"] != "complete" {
				continue
			}
			if _, err := lib.Scan(); err != nil {
				log.Warn("post-download scan failed", "err", err)
			} else {
				_ = settings.Set("library.metadata_schema", models.MetadataSchemaVersion)
			}
		}
	}()

	chatEP := &chatEndpointAdapter{im: im}
	chatSvc := chat.NewService(db.DB, hub, chatEP)
	chatSvc.SetStreaming(settings.Get("chat.streaming", "1") != "0")
	chatSvc.SetAttachDir(layout.Attachments)

	proxyEP := &proxyEndpointAdapter{im: im}
	px := proxy.NewServer(proxyEP, db.DB, logs.Logger("proxy", slog.LevelInfo).Logger)
	if err := px.LoadProfile(); err != nil {
		log.Warn("loading server profile failed", "err", err)
	}
	hostIt := hostit.NewBridge(settings, logs.Logger("hostit", slog.LevelInfo).Logger)
	if px.Config().Autostart {
		if err := px.Start(); err != nil {
			log.Warn("autostarting public API failed", "err", err)
		} else if err := hostIt.SyncTimeout(px.Config().Port, true); err != nil {
			log.Warn("hostit sync on autostart failed", "err", err)
		}
	}

	// Control API.
	srv := api.NewServer(auth.Token(*tokenFlag), hub, logs.Logger("api", slog.LevelInfo).Logger)
	srv.RegisterRoutes(&api.Deps{
		Hub: hub, Layout: layout, DB: db, Settings: settings, HF: hf, DL: dl,
		RT: rt, Lib: lib, IM: im, Chat: chatSvc, Proxy: px, HostIt: hostIt, Logs: logs, Quant: qm,
	})
	if err := srv.Start(*portFlag); err != nil {
		fmt.Fprintf(os.Stderr, `{"ready":false,"error":%q}`+"\n", "bind: "+err.Error())
		os.Exit(1)
	}

	// Structured readiness message consumed by the desktop launcher.
	ready, _ := json.Marshal(map[string]any{
		"ready": true, "port": srv.BoundPort(), "pid": os.Getpid(),
		"version": version.Version(), "commit": version.Commit,
	})
	fmt.Println(string(ready))

	if *selfTest {
		log.Info("selftest complete")
		return
	}

	// Parent-death watchdog: exit when the launcher disappears so inference
	// processes (and this backend) are never abandoned.
	if *ppidFlag > 0 {
		go watchParent(*ppidFlag, log, func() {
			shutdown(log, im, px, hostIt, srv)
			os.Exit(0)
		})
	}

	// Signal handling for standalone/dev usage.
	sigCh := make(chan os.Signal, 2)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	shutdown(log, im, px, hostIt, srv)
}

func shutdown(log *diagnostics.Logger, im *instances.Manager, px *proxy.Server, hi *hostit.Bridge, srv *api.Server) {
	log.Info("shutting down")
	if hi != nil {
		_ = hi.SyncTimeout(0, false)
	}
	px.Stop()
	im.StopAll()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx)
}

// watchParent polls parent liveness (kill(pid,0) on unix; OpenProcess on
// Windows via the platform file).
func watchParent(ppid int, log *diagnostics.Logger, onDead func()) {
	for {
		if !parentAlive(ppid) {
			log.Info("parent process gone; exiting", "ppid", ppid)
			onDead()
			return
		}
		time.Sleep(2 * time.Second)
	}
}
