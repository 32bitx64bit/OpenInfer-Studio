package mediagen

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/models"
	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

func TestWorkflowCacheIdentityTracksAutoPairedComponentsAndAPIDefaults(t *testing.T) {
	doc := decodeCapabilityFixture(t, `{"supported_modes":["img_gen"],"defaults_by_mode":{"img_gen":{"width":512}}}`)
	f := newCapabilitySD(t, doc, nil)
	m := newJobManager(t, f)
	root := t.TempDir()
	primary := filepath.Join(root, "qwen-image-2.1.gguf")
	vae := filepath.Join(root, "vae", "qwen_image_vae.safetensors")
	encoder := filepath.Join(root, "text_encoders", "qwen3vl_8b.safetensors")
	exe := filepath.Join(root, "runtime", "sd-server")
	help := sdHelpSample + "\n --llm FNAME encoder\n"
	for _, path := range []string{primary, vae, encoder, exe} {
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("fixture"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(filepath.Dir(exe), "help.txt"), []byte(help), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := m.db.Exec(`INSERT INTO runtimes(id,source,installed_at,install_dir,executable_path,version_output) VALUES(?,?,?,?,?,?)`, "rt", "custom-import", now(), filepath.Dir(exe), exe, "sd.cpp commit fixture"); err != nil {
		t.Fatal(err)
	}
	if _, err := m.db.Exec(`INSERT INTO models(id,alias,primary_path,created_at,updated_at) VALUES(?,?,?,?,?)`, "model-1", "qwen", primary, now(), now()); err != nil {
		t.Fatal(err)
	}
	m.rt = runtimes.NewManager(m.db, filepath.Dir(exe), nil, nil, nil)
	m.lib = models.NewLibrary(m.db, root, nil, nil)
	rt, err := m.rt.Get("rt")
	if err != nil {
		t.Fatal(err)
	}
	_, args, resolved, _, _, _, err := PrepareLaunch(rt, help, primary, LoadSettings{})
	if err != nil {
		t.Fatal(err)
	}
	if resolved.VAE != vae || resolved.LLM != encoder {
		t.Fatalf("auto-paired components missing: %+v", resolved)
	}
	files, err := cacheLaunchFiles(exe, args)
	if err != nil {
		t.Fatal(err)
	}
	sv := m.servers["model-1"]
	sv.runtimeID = "rt"
	sv.exe = exe
	sv.modelPath = primary
	sv.help = help
	sv.resolved = resolved
	sv.cacheSnapshot = &launchCacheSnapshot{Settings: resolved, Args: cacheArgs(args), SourceFiles: files, LoadedFiles: files}
	first, err := m.WorkflowCacheIdentity("model-1", nil)
	if err != nil || first == "" {
		t.Fatalf("identity=%q err=%v", first, err)
	}
	again, err := m.WorkflowCacheIdentity("model-1", nil)
	if err != nil || again != first {
		t.Fatalf("stable launch missed cache: %s %v", again, err)
	}
	doc["defaults_by_mode"].(map[string]any)["img_gen"].(map[string]any)["width"] = float64(768)
	changed, err := m.WorkflowCacheIdentity("model-1", nil)
	if err != nil || changed == first {
		t.Fatalf("API defaults did not invalidate: %s %v", changed, err)
	}
	if _, err := m.WorkflowCacheIdentity("model-1", &LoadOverrides{VAE: "/different"}); err == nil {
		t.Fatal("different launch override accepted")
	}
	for _, path := range []string{vae, encoder, exe, primary} {
		before, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		st, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("changed fixture"), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := m.WorkflowCacheIdentity("model-1", nil); err == nil {
			t.Fatalf("changed loaded file accepted: %s", path)
		}
		if err := os.WriteFile(path, before, 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(path, st.ModTime(), st.ModTime()); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCacheFileIdentityIncludesCatalogContentsAndExcludesSecrets(t *testing.T) {
	root := t.TempDir()
	exe := filepath.Join(root, "sd-server")
	loraDir := filepath.Join(root, "loras")
	if err := os.WriteFile(exe, []byte("runtime"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(loraDir, 0700); err != nil {
		t.Fatal(err)
	}
	args := []string{"--lora-model-dir", loraDir, "--api-key", "secret", "--listen-port", "1234"}
	before, err := cacheLaunchFiles(exe, args)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(loraDir, "film.safetensors"), []byte("weights"), 0600); err != nil {
		t.Fatal(err)
	}
	after, err := cacheLaunchFiles(exe, args)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(before, after) {
		t.Fatal("catalog contents excluded from identity")
	}
	if got := cacheArgs(args); !reflect.DeepEqual(got, []string{"--lora-model-dir", loraDir}) {
		t.Fatalf("ephemeral key/port in cache args: %v", got)
	}
	m := NewManager(nil, nil, nil, nil, nil, nil)
	if _, err := m.WorkflowCacheIdentity("unknown", nil); err == nil {
		t.Fatal("unknown cache identity accepted")
	}
}
