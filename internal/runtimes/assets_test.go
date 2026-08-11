package runtimes

import (
	"os"
	"path/filepath"
	"testing"
)

func TestClassifyAsset(t *testing.T) {
	cases := []struct {
		name           string
		platform, arch string
		backend        string
	}{
		{"llama-b6801-bin-ubuntu-x64.zip", "linux", "amd64", BackendCPU},
		{"llama-b6801-bin-ubuntu-vulkan-x64.zip", "linux", "amd64", BackendVulkan},
		{"llama-b6801-ubuntu-cuda-12-x64.zip", "linux", "amd64", BackendCUDA},
		{"llama-b6801-bin-win-cuda-12.4-x64.zip", "windows", "amd64", BackendCUDA},
		{"llama-b6801-bin-win-hip-x64-gfx1100.zip", "windows", "amd64", BackendHIP},
		{"llama-b6801-bin-macos-arm64.zip", "darwin", "arm64", BackendMetal},
		{"llama-b6801-bin-macos-x64.zip", "darwin", "amd64", BackendMetal},
		{"llama-b6801-bin-win-x64.zip", "windows", "amd64", BackendCPU},
		{"llama-b6801-bin-ubuntu-sycl-x64.zip", "linux", "amd64", BackendSYCL},
	}
	for _, c := range cases {
		p, a, b := ClassifyAsset(c.name)
		if p != c.platform || a != c.arch || b != c.backend {
			t.Errorf("ClassifyAsset(%q) = (%q,%q,%q), want (%q,%q,%q)",
				c.name, p, a, b, c.platform, c.arch, c.backend)
		}
	}
}

func TestDetectBackend(t *testing.T) {
	vulkanVersion := `load_backend: loaded Vulkan backend from /app/libggml-vulkan.so
load_backend: loaded CPU backend from /app/libggml-cpu-haswell.so
version: 9144 (4c1c3ac09)
built with GNU 15.2.0 for Linux x86_64`
	cudaVersion := `load_backend: loaded CUDA backend from /opt/libggml-cuda.so
load_backend: loaded CPU backend from /opt/libggml-cpu-x64.so
version: 1`

	cases := []struct {
		name    string
		version string
		hints   []string
		want    string
	}{
		{"version vulkan", vulkanVersion, nil, BackendVulkan},
		{"version cuda", cudaVersion, nil, BackendCUDA},
		{"filename vulkan", "version: 1\n", []string{"llama-b1-bin-ubuntu-vulkan-x64.zip"}, BackendVulkan},
		{"filename cpu", "version: 1\n", []string{"llama-b1-bin-ubuntu-x64.zip"}, BackendCPU},
		{"cuda beats vulkan in version", "loaded CUDA backend\nloaded Vulkan backend\n", nil, BackendCUDA},
		{"empty defaults cpu", "", nil, BackendCPU},
	}
	for _, c := range cases {
		if got := DetectBackend(c.version, c.hints); got != c.want {
			t.Errorf("%s: DetectBackend = %q, want %q", c.name, got, c.want)
		}
	}

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "libggml-hip.so"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := DetectBackend("version: 1", nil, dir); got != BackendHIP {
		t.Errorf("lib scan: got %q, want hip", got)
	}
}

func TestResolveAssetsPrefersCUDAForNVIDIA(t *testing.T) {
	rel := Release{Tag: "b6801", Assets: []Asset{
		{Name: "llama-b6801-bin-ubuntu-x64.zip"},
		{Name: "llama-b6801-bin-ubuntu-vulkan-x64.zip"},
		{Name: "llama-b6801-ubuntu-cuda-12-x64.zip"},
		{Name: "llama-b6801-bin-win-x64.zip"},
		{Name: "llama-b6801-bin-macos-arm64.zip"},
	}}
	m := MachineProfile{OS: "linux", Arch: "amd64", CUDA: true, Vulkan: true, GPUVendor: "nvidia"}
	matches := ResolveAssets(rel, m, "")
	if len(matches) == 0 {
		t.Fatal("no matches")
	}
	if matches[0].Backend != BackendCUDA {
		t.Errorf("top match = %s, want cuda; all: %+v", matches[0].Backend, matches)
	}
	// Windows/macOS assets must be excluded.
	for _, mt := range matches {
		p, _, _ := ClassifyAsset(mt.Asset.Name)
		if p != "" && p != "linux" {
			t.Errorf("foreign platform asset not filtered: %s", mt.Asset.Name)
		}
	}
}

func TestResolveAssetsPrefersMetalForAppleSilicon(t *testing.T) {
	rel := Release{Tag: "b6801", Assets: []Asset{
		{Name: "llama-b6801-bin-macos-arm64.zip"},
		{Name: "llama-b6801-bin-macos-x64.zip"},
	}}
	m := MachineProfile{OS: "darwin", Arch: "arm64", Metal: true, GPUVendor: "apple"}
	matches := ResolveAssets(rel, m, "")
	if len(matches) != 1 || matches[0].Backend != BackendMetal {
		t.Fatalf("want single metal match, got %+v", matches)
	}
}

func TestResolveAssetsUserPreference(t *testing.T) {
	rel := Release{Tag: "b6801", Assets: []Asset{
		{Name: "llama-b6801-bin-ubuntu-x64.zip"},
		{Name: "llama-b6801-bin-ubuntu-vulkan-x64.zip"},
	}}
	m := MachineProfile{OS: "linux", Arch: "amd64", Vulkan: true, GPUVendor: "amd"}
	matches := ResolveAssets(rel, m, BackendVulkan)
	if matches[0].Backend != BackendVulkan {
		t.Errorf("user-preferred vulkan not first: %+v", matches)
	}
	// CPU must remain listed as fallback.
	if matches[len(matches)-1].Backend != BackendCPU {
		t.Errorf("cpu fallback missing: %+v", matches)
	}
}

func TestStrongestVendor(t *testing.T) {
	if got := StrongestVendor([]string{"intel", "nvidia", "amd"}); got != "nvidia" {
		t.Errorf("got %s", got)
	}
	if got := StrongestVendor(nil); got != "" {
		t.Errorf("got %q", got)
	}
}

var _ = runtimeGOOS // referenced for symmetry; platform comes from MachineProfile in tests
