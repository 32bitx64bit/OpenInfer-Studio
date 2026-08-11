package runtimes

import (
	"io/fs"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

// Backend identifiers used across the app.
const (
	BackendCPU    = "cpu"
	BackendVulkan = "vulkan"
	BackendCUDA   = "cuda"
	BackendHIP    = "hip"
	BackendMetal  = "metal"
	BackendSYCL   = "sycl"
)

// osTokens maps platform → tokens that may appear in asset names. Release
// naming has changed over time, so matching is token-based rather than one
// fixed convention.
var osTokens = map[string][]string{
	"linux":   {"ubuntu", "linux"},
	"windows": {"win", "windows"},
	"darwin":  {"macos", "mac", "osx", "darwin"},
}

var archTokens = map[string][]string{
	"amd64": {"x64", "x86_64", "amd64"},
	"arm64": {"arm64", "aarch64"},
	"386":   {"x86", "i686", "i386"},
}

// backendTokens are matched in order of specificity (longest tokens first to
// avoid "cuda" matching inside "cuda-cu12").
var backendTokens = []struct {
	Backend string
	Tokens  []string
}{
	{BackendCUDA, []string{"cuda"}},
	{BackendHIP, []string{"hip", "rocm"}},
	{BackendVulkan, []string{"vulkan"}},
	{BackendMetal, []string{"metal"}},
	{BackendSYCL, []string{"sycl"}},
	{BackendCPU, []string{"cpu"}},
}

// tokenize lowercases and splits an asset name on non-alphanumerics.
func tokenize(name string) []string {
	return strings.FieldsFunc(strings.ToLower(name), func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '.' || r == '_')
	})
}

func containsAny(tokens []string, candidates []string) bool {
	for _, t := range tokens {
		for _, c := range candidates {
			if t == c || strings.Contains(t, c) {
				return true
			}
		}
	}
	return false
}

// DetectBackend infers the primary acceleration backend for a custom or
// unknown build. Evidence is combined from llama-server --version output
// (load_backend lines / libggml-* paths), colocated libraries under
// searchRoots, and path name hints (archive or executable names).
//
// CPU is always present in modern builds; when a GPU backend is detected it
// wins. Among GPU backends, priority matches backendTokens (CUDA > HIP >
// Vulkan > Metal > SYCL).
func DetectBackend(versionOut string, pathHints []string, searchRoots ...string) string {
	found := map[string]bool{}
	collectBackendsFromText(versionOut, found)
	for _, root := range searchRoots {
		if root == "" {
			continue
		}
		collectBackendsFromLibs(root, found)
	}
	for _, hint := range pathHints {
		if hint == "" {
			continue
		}
		_, _, b := ClassifyAsset(hint)
		if b != "" {
			found[b] = true
		}
	}
	for _, bt := range backendTokens {
		if bt.Backend == BackendCPU {
			continue
		}
		if found[bt.Backend] {
			return bt.Backend
		}
	}
	return BackendCPU
}

func collectBackendsFromText(text string, found map[string]bool) {
	if text == "" {
		return
	}
	lower := strings.ToLower(text)
	for _, bt := range backendTokens {
		if bt.Backend == BackendCPU {
			continue
		}
		for _, tok := range bt.Tokens {
			if strings.Contains(lower, "libggml-"+tok) ||
				strings.Contains(lower, "ggml-"+tok) ||
				strings.Contains(lower, "loaded "+tok+" backend") ||
				strings.Contains(lower, "ggml_"+tok+":") {
				found[bt.Backend] = true
			}
		}
	}
}

// collectBackendsFromLibs scans a tree for ggml backend shared libraries
// (libggml-vulkan.so, ggml-cuda.dll, …). Depth is capped so import stays cheap.
func collectBackendsFromLibs(root string, found map[string]bool) {
	const maxDepth = 4
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		rel, relErr := filepath.Rel(root, p)
		if relErr != nil {
			return nil
		}
		depth := 0
		if rel != "." {
			depth = strings.Count(rel, string(filepath.Separator)) + 1
		}
		if d.IsDir() {
			if depth > maxDepth {
				return filepath.SkipDir
			}
			return nil
		}
		name := strings.ToLower(d.Name())
		if !strings.Contains(name, "ggml") {
			return nil
		}
		for _, bt := range backendTokens {
			if bt.Backend == BackendCPU {
				continue
			}
			for _, tok := range bt.Tokens {
				if strings.Contains(name, "ggml-"+tok) {
					found[bt.Backend] = true
					break
				}
			}
		}
		return nil
	})
}

// ClassifyAsset extracts platform, architecture and backend from an asset
// name. Unknown fields are empty strings.
func ClassifyAsset(name string) (platform, arch, backend string) {
	tokens := tokenize(name)
	lower := strings.ToLower(name)
	for osName, toks := range osTokens {
		for _, t := range toks {
			if strings.Contains(lower, t) {
				platform = osName
				break
			}
		}
	}
	for archName, toks := range archTokens {
		if containsAny(tokens, toks) {
			arch = archName
			break
		}
	}
	for _, bt := range backendTokens {
		if containsAny(tokens, bt.Tokens) {
			backend = bt.Backend
			break
		}
	}
	// macOS releases are Metal-capable by default; plain platform archives
	// with no backend token are CPU builds elsewhere.
	if backend == "" {
		if platform == "darwin" {
			backend = BackendMetal
		} else if platform != "" {
			backend = BackendCPU
		}
	}
	return platform, arch, backend
}

// AssetMatch pairs an asset with a score for the current machine.
type AssetMatch struct {
	Asset   Asset  `json:"asset"`
	Backend string `json:"backend"`
	Score   int    `json:"score"`
	Reason  string `json:"reason"`
}

// hw describes the machine for resolution (subset of hardware.Info to keep
// runtimes decoupled).
type MachineProfile struct {
	OS, Arch  string
	Vulkan    bool
	CUDA      bool
	HIP       bool
	Metal     bool
	SYCL      bool
	GPUVendor string // strongest detected vendor: nvidia|amd|intel|apple|""
}

// ResolveAssets scores all assets of a release for this machine and a
// preferred backend ("" = automatic). Higher score is better; negative means
// incompatible. Results are sorted best-first.
func ResolveAssets(rel Release, m MachineProfile, preferBackend string) []AssetMatch {
	var out []AssetMatch
	for _, a := range rel.Assets {
		lower := strings.ToLower(a.Name)
		// Only binary archives; skip source tarballs and cudart bundles.
		if !(strings.HasSuffix(lower, ".zip") || strings.HasSuffix(lower, ".tar.gz")) {
			continue
		}
		if strings.Contains(lower, "cudart") || strings.Contains(lower, "src") {
			continue
		}
		platform, arch, backend := ClassifyAsset(a.Name)
		score := 0
		reasons := []string{}
		if platform != "" && platform != m.OS {
			continue // wrong OS
		}
		if platform == m.OS {
			score += 10
			reasons = append(reasons, "OS match")
		}
		if arch != "" && arch != m.Arch {
			continue
		}
		if arch == m.Arch {
			score += 10
			reasons = append(reasons, "architecture match")
		}

		avail := map[string]bool{
			BackendCPU: true, BackendVulkan: m.Vulkan, BackendCUDA: m.CUDA,
			BackendHIP: m.HIP, BackendMetal: m.Metal, BackendSYCL: m.SYCL,
		}
		if !avail[backend] {
			// Backend runtime not detected: still list it, but ranked low.
			score -= 5
			reasons = append(reasons, "backend runtime not detected")
		}
		if preferBackend != "" {
			if backend == preferBackend {
				score += 40
				reasons = append(reasons, "user-preferred backend")
			} else if backend == BackendCPU {
				score += 1 // always a fallback
			}
		} else {
			// Automatic preference order for the detected vendor.
			score += autoBackendScore(backend, m)
		}
		out = append(out, AssetMatch{Asset: a, Backend: backend, Score: score, Reason: strings.Join(reasons, "; ")})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	return out
}

func autoBackendScore(backend string, m MachineProfile) int {
	switch m.GPUVendor {
	case "apple":
		if backend == BackendMetal {
			return 30
		}
	case "nvidia":
		switch backend {
		case BackendCUDA:
			return 30
		case BackendVulkan:
			return 20
		}
	case "amd":
		if m.OS == "linux" {
			switch backend {
			case BackendHIP:
				return 30
			case BackendVulkan:
				return 25
			}
		} else {
			switch backend {
			case BackendVulkan:
				return 28
			case BackendHIP:
				return 25
			}
		}
	case "intel":
		switch backend {
		case BackendVulkan:
			return 28
		case BackendSYCL:
			return 25
		}
	}
	if backend == BackendCPU {
		return 5
	}
	return 0
}

// StrongestVendor reduces a GPU list to the vendor with the best acceleration
// story (nvidia > amd > intel > apple handled by OS).
func StrongestVendor(vendors []string) string {
	pri := map[string]int{"nvidia": 4, "amd": 3, "intel": 2, "apple": 1}
	best, bestScore := "", -1
	for _, v := range vendors {
		if pri[v] > bestScore {
			best, bestScore = v, pri[v]
		}
	}
	return best
}

// runtimeGOOS is a small helper for tests overriding the platform.
func runtimeGOOS() string { return runtime.GOOS }
