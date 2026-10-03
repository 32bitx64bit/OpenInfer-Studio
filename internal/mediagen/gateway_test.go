package mediagen

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/openinfer/openinfer-studio/internal/models"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

func gatewayForSD(t *testing.T, native *httptest.Server) (*sdGateway, int) {
	t.Helper()
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	gateway, err := startSDGateway(listener, (&fakeSD{srv: native}).port(t), "test-key")
	if err != nil {
		_ = listener.Close()
		t.Fatal(err)
	}
	t.Cleanup(gateway.Close)
	return gateway, port
}

func gatewayRequest(t *testing.T, client *http.Client, port int, method, path string, headers http.Header, body string) *http.Response {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	t.Cleanup(cancel)
	req, err := http.NewRequestWithContext(ctx, method, fmt.Sprintf("http://127.0.0.1:%d%s", port, path), strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header = headers.Clone()
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

func TestSDGatewayAuthAndNativeForwarding(t *testing.T) {
	type forwarded struct{ method, path, query, host, auth, proxyAuth, cookie, forwarded, body string }
	requests := make(chan forwarded, 8)
	native := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		requests <- forwarded{r.Method, r.URL.Path, r.URL.RawQuery, r.Host, r.Header.Get("Authorization"), r.Header.Get("Proxy-Authorization"), r.Header.Get("Cookie"), r.Header.Get("X-Forwarded-Host"), string(body)}
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			_, _ = io.WriteString(w, `{"current_mode":"img_gen","supported_modes":["img_gen"],"features_by_mode":{"img_gen":{"mask_image":true}},"samplers":["euler"],"schedulers":["discrete"]}`)
		} else {
			w.WriteHeader(http.StatusAccepted)
			_, _ = io.WriteString(w, `{"id":"native-job"}`)
		}
	}))
	t.Cleanup(native.Close)
	gateway, port := gatewayForSD(t, native)
	client := sdHTTPClient()
	t.Cleanup(client.CloseIdleConnections)
	for _, headers := range []http.Header{
		{}, {"Authorization": {"Bearer wrong"}}, {"Authorization": {"test-key"}},
		{"Authorization": {"Bearer test-key", "Bearer test-key"}},
	} {
		resp := gatewayRequest(t, client, port, http.MethodGet, "/sdcpp/v1/capabilities", headers, "")
		if resp.StatusCode != http.StatusUnauthorized {
			t.Fatalf("unauthenticated status=%d", resp.StatusCode)
		}
		resp.Body.Close()
	}
	for _, method := range []string{http.MethodConnect, http.MethodGet} {
		headers := http.Header{"Authorization": {"Bearer test-key"}}
		if method == http.MethodGet {
			headers.Set("Upgrade", "websocket")
		}
		resp := gatewayRequest(t, client, port, method, "/sdcpp/v1/capabilities", headers, "")
		if resp.StatusCode != http.StatusMethodNotAllowed {
			t.Fatalf("tunnel/upgrade accepted: %d", resp.StatusCode)
		}
		resp.Body.Close()
	}
	select {
	case <-requests:
		t.Fatal("rejected request reached native server")
	default:
	}
	m := NewManager(nil, nil, nil, nil, nil, nil)
	m.servers["model"] = &server{modelID: "model", port: port, nativePort: (&fakeSD{srv: native}).port(t), apiKey: "test-key", gateway: gateway, ready: true, state: ServerReady}
	caps, err := m.GenerationCapabilities("model")
	if err != nil || !caps.Known || !caps.SupportsFeature("img_gen", "mask_image") {
		t.Fatalf("capabilities not forwarded: caps=%+v err=%v", caps, err)
	}
	capRequest := <-requests
	if capRequest.auth != "" || capRequest.path != "/sdcpp/v1/capabilities" {
		t.Fatalf("native capability request=%+v", capRequest)
	}
	req, _ := http.NewRequest(http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/sdcpp/v1/img_gen?mode=test", port), strings.NewReader(`{"prompt":"fixture"}`))
	req.Host = "untrusted.invalid"
	req.Header.Set("Proxy-Authorization", "Bearer test-key")
	req.Header.Set("Cookie", "private=test-key")
	req.Header.Set("X-Forwarded-Host", "untrusted.invalid")
	resp, err := m.doSD(req, port)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var result map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil || result["id"] != "native-job" || resp.StatusCode != http.StatusAccepted {
		t.Fatalf("native response not preserved: %v %v", result, err)
	}
	got := <-requests
	if got.method != http.MethodPost || got.query != "mode=test" || got.body != `{"prompt":"fixture"}` || got.auth != "" || got.proxyAuth != "" || got.cookie != "" || got.forwarded != "" || got.host != strings.TrimPrefix(native.URL, "http://") {
		t.Fatalf("native request=%+v", got)
	}
	nativePort := (&fakeSD{srv: native}).port(t)
	rawReq, _ := http.NewRequest(http.MethodGet, native.URL+"/sdcpp/v1/capabilities", nil)
	if _, err := m.doSD(rawReq, nativePort); !errors.Is(err, ErrServerNotRunning) {
		t.Fatalf("manager reached raw native port: %v", err)
	}
}

func TestSDGatewayRefusesRedirectsAndEnvironmentProxies(t *testing.T) {
	var otherCalls atomic.Int32
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { otherCalls.Add(1) }))
	t.Cleanup(other.Close)
	t.Setenv("HTTP_PROXY", other.URL)
	t.Setenv("HTTPS_PROXY", other.URL)
	t.Setenv("ALL_PROXY", other.URL)
	native := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, other.URL, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(native.Close)
	gateway, port := gatewayForSD(t, native)
	client := sdHTTPClient()
	t.Cleanup(client.CloseIdleConnections)
	resp := gatewayRequest(t, client, port, http.MethodPost, "/sdcpp/v1/img_gen", http.Header{"Authorization": {"Bearer test-key"}}, "{}")
	if resp.StatusCode != http.StatusBadGateway || resp.Header.Get("Location") != "" || otherCalls.Load() != 0 || gateway.transport.Proxy != nil || client.Transport.(*http.Transport).Proxy != nil {
		t.Fatal("gateway redirected or used an environment proxy")
	}
}

func TestSDGatewayCancellationAndClosure(t *testing.T) {
	for _, closeGateway := range []bool{false, true} {
		t.Run(fmt.Sprintf("close=%t", closeGateway), func(t *testing.T) {
			started, canceled := make(chan struct{}), make(chan struct{})
			native := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				close(started)
				<-r.Context().Done()
				close(canceled)
			}))
			t.Cleanup(native.Close)
			gateway, port := gatewayForSD(t, native)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			req, _ := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/sdcpp/v1/upscale", port), strings.NewReader("{}"))
			req.Header.Set("Authorization", "Bearer test-key")
			client := sdHTTPClient()
			t.Cleanup(client.CloseIdleConnections)
			finished := make(chan error, 1)
			go func() {
				resp, err := client.Do(req)
				if resp != nil {
					resp.Body.Close()
				}
				finished <- err
			}()
			select {
			case <-started:
			case <-time.After(2 * time.Second):
				t.Fatal("native request did not start")
			}
			if closeGateway {
				gateway.Close()
			} else {
				cancel()
			}
			select {
			case <-canceled:
			case <-time.After(2 * time.Second):
				t.Fatal("native HTTP request was not canceled")
			}
			select {
			case err := <-finished:
				if err == nil {
					t.Fatal("canceled request succeeded")
				}
			case <-time.After(2 * time.Second):
				t.Fatal("public request did not finish")
			}
			gateway.Close()
			gateway.Close()
			assertGatewayClosed(t, gateway)
		})
	}
}

func assertGatewayClosed(t *testing.T, gateway *sdGateway) {
	t.Helper()
	select {
	case <-gateway.done:
	case <-time.After(time.Second):
		t.Fatal("gateway serve loop did not close")
	}
	conn, err := net.DialTimeout("tcp4", gateway.listener.Addr().String(), 100*time.Millisecond)
	if err == nil {
		conn.Close()
		t.Fatal("closed gateway still accepts connections")
	}
}

func TestSDGatewayLifecycleAndClaimShutdown(t *testing.T) {
	for _, action := range []string{"stop", "abort", "crash", "replace", "stopped claim", "canceled claim"} {
		t.Run(action, func(t *testing.T) {
			native := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, "{}") }))
			t.Cleanup(native.Close)
			gateway, port := gatewayForSD(t, native)
			m := NewManager(nil, nil, nil, nil, nil, nil)
			ctx, cancel := context.WithCancel(context.Background())
			sv := &server{modelID: "model", port: port, gateway: gateway, apiKey: "test-key", ready: true, state: ServerReady, launchCtx: ctx, cancel: cancel}
			m.servers["model"] = sv
			switch action {
			case "stop":
				m.StopServer("model")
			case "abort":
				m.abortServer("model", port, "fixture abort")
			case "crash":
				m.handleServerExit("model", sv)
			case "replace":
				replacementGateway, replacementPort := gatewayForSD(t, native)
				m.servers["model"] = &server{modelID: "model", port: replacementPort, gateway: replacementGateway, ready: true, state: ServerReady}
				m.closeSDServer(sv)
				m.handleServerExit("model", sv)
				select {
				case <-replacementGateway.done:
					t.Fatal("old launch closed replacement gateway")
				default:
				}
			case "stopped claim", "canceled claim":
				sv.state = ServerStarting
				if action == "stopped claim" {
					m.StopServer("model")
				} else {
					cancel()
				}
				if err := m.attachSDTransport(sv, &sdEndpoint{port: port, key: "test-key", gateway: gateway}); !errors.Is(err, context.Canceled) {
					t.Fatalf("dead claim accepted endpoint: %v", err)
				}
				gateway.Close()
			}
			assertGatewayClosed(t, gateway)
		})
	}
}

func TestSDGatewayStopCancelsReadinessRequest(t *testing.T) {
	started, canceled := make(chan struct{}), make(chan struct{})
	native := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		<-r.Context().Done()
		close(canceled)
	}))
	t.Cleanup(native.Close)
	gateway, port := gatewayForSD(t, native)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m := NewManager(nil, nil, nil, nil, nil, nil)
	sv := &server{modelID: "model", port: port, gateway: gateway, apiKey: "test-key", state: ServerStarting, launchCtx: ctx, cancel: cancel}
	m.servers["model"] = sv
	finished := make(chan error, 1)
	go func() { finished <- m.waitReady(sv, make(chan struct{}), time.Minute) }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("readiness request did not start")
	}
	m.StopServer("model")
	select {
	case err := <-finished:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("stopped readiness returned %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("stop did not interrupt readiness")
	}
	select {
	case <-canceled:
	case <-time.After(time.Second):
		t.Fatal("stopped readiness still reached native runtime")
	}
	assertGatewayClosed(t, gateway)
}

func TestSDTransportStockFlagsAndNativeAuth(t *testing.T) {
	model := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(model, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, extra string
		native      bool
	}{
		{"stock", "", false},
		{"key file only", "\n --api-key-file FILE\n", false},
		{"prose", "\nAuthentication is possible with --api-key TOKEN\n", false},
		{"other option description", "\n --other TEXT refers to --api-key\n", false},
		{"removed", "\n --api-key TOKEN has been removed\n", false},
		{"native", "\n -k, --api-key TOKEN\n", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			help := sdHelpSample + tc.extra
			base, err := BuildServerArgs(LoadSettings{}, model, false, ParseSDCapabilities(help), help, "127.0.0.1", 0)
			if err != nil {
				t.Fatal(err)
			}
			args, endpoint, err := configureSDTransport(help, base)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(endpoint.close)
			if (endpoint.gateway == nil) != tc.native || len(endpoint.key) != 64 || endpoint.port == 0 || endpoint.nativePort == 0 || (endpoint.port == endpoint.nativePort) != tc.native {
				t.Fatal("incorrect native/gateway endpoint selection")
			}
			if strings.Contains(strings.Join(args, " "), "--api-key") != tc.native || !reflect.DeepEqual(cacheArgs(args), cacheArgs(base)) {
				t.Fatal("unsupported flag or ephemeral transport details in cache args")
			}
			if strings.Contains(strings.Join(redactSDArgs(args), " "), endpoint.key) {
				t.Fatal("API key present in logged args")
			}
			if !tc.native {
				return
			}
			listener, err := net.Listen("tcp4", fmt.Sprintf("127.0.0.1:%d", endpoint.nativePort))
			if err != nil {
				t.Fatal(err)
			}
			native := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") != "Bearer "+endpoint.key {
					http.Error(w, "unauthorized", http.StatusUnauthorized)
					return
				}
				_, _ = io.WriteString(w, `{"supported_modes":["img_gen"]}`)
			}))
			native.Listener.Close()
			native.Listener = listener
			native.Start()
			t.Cleanup(native.Close)
			m := NewManager(nil, nil, nil, nil, nil, nil)
			m.servers["model"] = &server{modelID: "model", port: endpoint.port, apiKey: endpoint.key, ready: true, state: ServerReady}
			caps, err := m.GenerationCapabilities("model")
			if err != nil || !caps.Known {
				t.Fatalf("authenticated direct capabilities failed: %v", err)
			}
		})
	}
}

func TestSDGatewayClosesAfterStartupFailure(t *testing.T) {
	f := newFakeSD(t, nil)
	m := newJobManager(t, f)
	m.StopServer("model-1")
	root := t.TempDir()
	exe, model := filepath.Join(root, "sd-server"), filepath.Join(root, "model.safetensors")
	for _, path := range []string{exe, model} {
		// An unexecutable fixture fails after endpoint allocation and never
		// loads weights or invokes an installed runtime.
		if err := os.WriteFile(path, []byte("fixture"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "help.txt"), []byte(sdHelpSample), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := m.db.Exec(`INSERT INTO runtimes(id,source,installed_at,install_dir,executable_path,version_output) VALUES(?,?,?,?,?,?)`, "rt", "custom-import", now(), root, exe, "fixture"); err != nil {
		t.Fatal(err)
	}
	if _, err := m.db.Exec(`INSERT INTO models(id,alias,primary_path,created_at,updated_at) VALUES(?,?,?,?,?)`, "model-1", "fixture", model, now(), now()); err != nil {
		t.Fatal(err)
	}
	m.rt = runtimes.NewManager(m.db, root, nil, nil, nil)
	m.lib = models.NewLibrary(m.db, root, nil, nil)
	if _, err := m.EnsureServer("model-1", LoadSettings{RuntimeID: "rt"}); err == nil {
		t.Fatal("unexecutable fixture launched")
	}
	sv := m.servers["model-1"]
	if sv.gateway == nil || sv.port != 0 || sv.ready || sv.state != ServerFailed {
		t.Fatalf("failed startup retained endpoint: gateway=%t port=%d state=%s", sv.gateway != nil, sv.port, sv.state)
	}
	assertGatewayClosed(t, sv.gateway)
}
