package mediagen

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

type cacheFile struct {
	Path     string
	Resolved string
	Size     int64
	Modified int64
	Mode     uint32
	FileID   string
}

type launchCacheSnapshot struct {
	Settings    LoadSettings
	Args        []string
	SourceFiles []cacheFile
	LoadedFiles []cacheFile
}

// WorkflowCacheIdentity is available only for a ready, unchanged launch with
// known loaded-model HTTP capabilities. It never loads/reloads a model. The
// parent must treat any error as a cache miss. No credentials enter the hash.
func (m *Manager) WorkflowCacheIdentity(modelID string, ov *LoadOverrides) (string, error) {
	if m.rt == nil || m.lib == nil {
		return "", fmt.Errorf("workflow cache identity requires managed runtime and model metadata")
	}
	m.mu.Lock()
	sv := m.servers[modelID]
	if sv == nil || !sv.ready || sv.cacheSnapshot == nil {
		m.mu.Unlock()
		return "", fmt.Errorf("workflow cache identity requires a known ready launch")
	}
	snapshot := *sv.cacheSnapshot
	runtimeID, help, exe, port := sv.runtimeID, sv.help, sv.exe, sv.port
	m.mu.Unlock()
	settings := snapshot.Settings
	if ov != nil && !ov.SatisfiedBy(settings) {
		return "", fmt.Errorf("workflow launch overrides would reload the model")
	}
	if strings.TrimSpace(settings.RawArgs) != "" {
		return "", fmt.Errorf("workflow cache identity unavailable for raw launch overrides")
	}
	rt, err := m.rt.Get(runtimeID)
	if err != nil {
		return "", err
	}
	if rt.VersionOutput == "" {
		return "", fmt.Errorf("runtime version identity is unknown")
	}
	mdl, err := m.lib.Get(modelID)
	if err != nil {
		return "", err
	}
	// Re-run companion pairing against the current model directory. This
	// catches newly discovered/replaced default VAEs and text encoders even
	// if the model's explicit settings did not change.
	launchSettings := settings
	m.mu.Lock()
	if m.servers[modelID] != sv {
		m.mu.Unlock()
		return "", ErrServerNotRunning
	}
	launchSettings = sv.settings
	m.mu.Unlock()
	preparedExe, args, resolved, preparedHelp, _, _, err := PrepareLaunch(rt, help, mdl.PrimaryPath, launchSettings)
	if err != nil {
		return "", err
	}
	if preparedExe != exe || preparedHelp != help || resolved != settings {
		return "", fmt.Errorf("effective prepared launch changed; cache identity unavailable until reload")
	}
	sourceFiles, err := cacheLaunchFiles(exe, args)
	if err != nil {
		return "", err
	}
	if !reflect.DeepEqual(sourceFiles, snapshot.SourceFiles) {
		return "", fmt.Errorf("checkpoint, runtime, or auto-paired component changed since model load")
	}
	loadedFiles, err := cacheLaunchFiles(exe, snapshot.Args)
	if err != nil {
		return "", err
	}
	if !reflect.DeepEqual(loadedFiles, snapshot.LoadedFiles) {
		return "", fmt.Errorf("loaded runtime or component files changed since model load")
	}
	helpNow, err := m.rt.HelpOutput(runtimeID)
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(helpNow) != "" && helpNow != help {
		return "", fmt.Errorf("runtime help identity changed")
	}
	apiDoc, err := m.Capabilities(context.Background(), modelID)
	if err != nil {
		return "", err
	}
	caps := normalizeAPICapabilities(apiDoc, modelID, runtimeID)
	if !caps.Known {
		return "", fmt.Errorf("workflow cache identity requires known loaded-model API capabilities: %s", caps.Reason)
	}
	identity := struct {
		Version                  int
		Runtime                  *runtimes.Runtime
		Settings                 LoadSettings
		Args                     []string
		Help                     string
		API                      *APICapabilities
		APIDocument              map[string]any
		SourceFiles, LoadedFiles []cacheFile
	}{Version: 1, Runtime: rt, Settings: settings, Args: cacheArgs(args), Help: help, API: caps, APIDocument: apiDoc, SourceFiles: sourceFiles, LoadedFiles: loadedFiles}
	// Runtime UI fields (Preferred, UsedByModels) do not affect execution.
	identity.Runtime = &runtimes.Runtime{ID: rt.ID, Source: rt.Source, ReleaseID: rt.ReleaseID, Build: rt.Build, Commit: rt.Commit, Backend: rt.Backend, ExecutablePath: exe, VersionOutput: rt.VersionOutput, ArchiveSHA256: rt.ArchiveSHA256}
	raw, err := json.Marshal(identity)
	if err != nil {
		return "", err
	}
	m.mu.Lock()
	unchanged := m.servers[modelID] == sv && sv.ready && sv.port == port && sv.cacheSnapshot != nil && reflect.DeepEqual(*sv.cacheSnapshot, snapshot)
	m.mu.Unlock()
	if !unchanged {
		return "", fmt.Errorf("diffusion launch changed while computing workflow cache identity")
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:]), nil
}

func cacheArgs(args []string) []string {
	out := []string{}
	for i := 0; i < len(args); i++ {
		if args[i] == "--api-key" || args[i] == "--listen-port" {
			i++
			continue
		}
		out = append(out, args[i])
	}
	return out
}

func cacheLaunchFiles(exe string, args []string) ([]cacheFile, error) {
	paths := map[string]bool{exe: false}
	fileFlags := map[string]bool{
		"--model": false, "--diffusion-model": false, "--high-noise-diffusion-model": false,
		"--vae": false, "--taesd": false, "--tae": false, "--control-net": false, "--ip-adapter": false,
		"--llm": false, "--llm_vision": false, "--t5xxl": false, "--clip_l": false, "--clip_g": false, "--clip_vision": false,
		"--tokenizer": false, "--motion-module": false, "--upscale-model": false,
		"--lora-model-dir": true, "--hires-upscalers-dir": true, "--embd-dir": true,
	}
	// Only the final occurrence of a component flag is effective.
	effective := map[string]string{}
	for i := 0; i+1 < len(args); i++ {
		if _, ok := fileFlags[args[i]]; ok {
			effective[args[i]] = args[i+1]
		}
	}
	for flag, path := range effective {
		paths[path] = fileFlags[flag]
	}
	var files []cacheFile
	for path, dir := range paths {
		file, err := statCacheFile(path)
		if err != nil {
			return nil, err
		}
		files = append(files, file)
		if dir {
			err = filepath.WalkDir(path, func(child string, d os.DirEntry, err error) error {
				if err != nil {
					return err
				}
				if child == path {
					return nil
				}
				if d.Type()&os.ModeSymlink != 0 {
					return fmt.Errorf("cache identity cannot verify symlinked component catalog")
				}
				if len(files) > 4096 {
					return fmt.Errorf("component catalog exceeds cache identity entry bound")
				}
				f, err := statCacheFile(child)
				if err != nil {
					return err
				}
				files = append(files, f)
				return nil
			})
			if err != nil {
				return nil, err
			}
		}
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	return files, nil
}

func statCacheFile(path string) (cacheFile, error) {
	if !filepath.IsAbs(path) {
		return cacheFile{}, fmt.Errorf("cache input path must be absolute")
	}
	real, err := filepath.EvalSymlinks(path)
	if err != nil {
		return cacheFile{}, err
	}
	st, err := os.Stat(real)
	if err != nil {
		return cacheFile{}, err
	}
	if !st.Mode().IsRegular() && !st.IsDir() {
		return cacheFile{}, fmt.Errorf("cache input must be a regular file or directory")
	}
	// Device/inode (where provided) detect replacement with preserved size
	// and mtime. Avoid atime, which changes when the model is read.
	fileID := ""
	v := reflect.ValueOf(st.Sys())
	if v.IsValid() && v.Kind() == reflect.Pointer && !v.IsNil() {
		v = v.Elem()
	}
	if v.IsValid() && v.Kind() == reflect.Struct {
		for _, name := range []string{"Dev", "Ino", "VolumeSerialNumber", "FileIndexHigh", "FileIndexLow"} {
			f := v.FieldByName(name)
			if f.IsValid() && f.CanInterface() {
				fileID += fmt.Sprintf("%s=%v;", name, f.Interface())
			}
		}
	}
	return cacheFile{Path: path, Resolved: real, Size: st.Size(), Modified: st.ModTime().UnixNano(), Mode: uint32(st.Mode()), FileID: fileID}, nil
}
