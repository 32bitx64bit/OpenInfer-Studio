// Package hardware detects CPU, memory, GPUs and acceleration APIs.
// Probes are independent: one failure does not block startup.
package hardware

import (
	"runtime"
)

// GPU describes one detected graphics device.
type GPU struct {
	Name   string `json:"name"`
	Vendor string `json:"vendor"` // nvidia|amd|intel|apple|unknown
	VRAM   uint64 `json:"vram_bytes"`
	Driver string `json:"driver,omitempty"`
}

// Probe records the outcome of one detection step.
type Probe struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail,omitempty"`
}

// Info is the full hardware report.
type Info struct {
	OS               string   `json:"os"`
	OSVersion        string   `json:"os_version"`
	Arch             string   `json:"arch"`
	CPUModel         string   `json:"cpu_model"`
	LogicalCores     int      `json:"logical_cores"`
	PhysicalCores    int      `json:"physical_cores"`
	CPUFeatures      []string `json:"cpu_features"`
	RAMTotal         uint64   `json:"ram_total"`
	RAMAvailable     uint64   `json:"ram_available"`
	GPUs             []GPU    `json:"gpus"`
	Vulkan           bool     `json:"vulkan"`
	CUDA             bool     `json:"cuda"`
	HIP              bool     `json:"hip"`
	Metal            bool     `json:"metal"`
	SYCL             bool     `json:"sycl"`
	DiskFreeModels   uint64   `json:"disk_free_models"`
	DiskFreeRuntimes uint64   `json:"disk_free_runtimes"`
	Probes           []Probe  `json:"probes"`
}

// Detect runs all platform probes. modelsDir/runtimesDir feed disk-space
// checks. It never returns an error; individual failures land in Probes.
func Detect(modelsDir, runtimesDir string) *Info {
	info := &Info{OS: runtime.GOOS, Arch: runtime.GOARCH, LogicalCores: runtime.NumCPU()}
	detectPlatform(info)
	info.DiskFreeModels = diskFree(modelsDir)
	info.DiskFreeRuntimes = diskFree(runtimesDir)
	return info
}

// DiskFree is the space available to an unprivileged writer on the volume
// holding dir, or on its nearest existing ancestor; 0 when it cannot be read.
func DiskFree(dir string) uint64 { return diskFree(dir) }

// Recommendation explains an automatic backend choice. The UI must display
// the reason and allow override — detection is a heuristic, not a guarantee.
type Recommendation struct {
	Backend      string   `json:"backend"`      // metal|cuda|vulkan|hip|sycl|cpu
	Reason       string   `json:"reason"`       // human explanation
	Alternatives []string `json:"alternatives"` // other viable backends
}

// RecommendBackend picks a backend from detected hardware. It never claims
// certainty: final validation is the runtime health check.
func RecommendBackend(i *Info) Recommendation {
	if i.OS == "darwin" && i.Arch == "arm64" && i.Metal {
		return Recommendation{Backend: "metal",
			Reason:       "Apple Silicon detected with Metal support",
			Alternatives: []string{"cpu"}}
	}
	hasNVIDIA, hasAMD, hasIntel := false, false, false
	for _, g := range i.GPUs {
		switch g.Vendor {
		case "nvidia":
			hasNVIDIA = true
		case "amd":
			hasAMD = true
		case "intel":
			hasIntel = true
		}
	}
	switch {
	case hasNVIDIA && i.CUDA:
		return Recommendation{Backend: "cuda",
			Reason:       "NVIDIA GPU with a working CUDA driver detected",
			Alternatives: alts(i.Vulkan, true, "vulkan", "cpu")}
	case hasNVIDIA && i.Vulkan:
		return Recommendation{Backend: "vulkan",
			Reason:       "NVIDIA GPU detected; CUDA not found, Vulkan is available",
			Alternatives: []string{"cpu", "cuda"}}
	case hasAMD && i.OS == "linux" && i.HIP:
		return Recommendation{Backend: "hip",
			Reason:       "AMD GPU with ROCm/HIP detected on Linux",
			Alternatives: alts(i.Vulkan, true, "vulkan", "cpu")}
	case hasAMD && i.Vulkan:
		return Recommendation{Backend: "vulkan",
			Reason:       "AMD GPU detected; Vulkan offers broad compatibility",
			Alternatives: alts(i.HIP, true, "hip", "cpu")}
	case hasIntel && i.Vulkan:
		return Recommendation{Backend: "vulkan",
			Reason:       "Intel GPU detected with Vulkan support",
			Alternatives: alts(i.SYCL, true, "sycl", "cpu")}
	case i.Vulkan:
		return Recommendation{Backend: "vulkan",
			Reason:       "Vulkan is available; no specific GPU vendor preference",
			Alternatives: []string{"cpu"}}
	default:
		return Recommendation{Backend: "cpu",
			Reason:       "No GPU acceleration detected; using portable CPU backend",
			Alternatives: []string{}}
	}
}

func alts(cond bool, alwaysCPU bool, opts ...string) []string {
	out := []string{}
	for _, o := range opts {
		if o == "cpu" && !alwaysCPU {
			continue
		}
		if (o == "vulkan" || o == "hip" || o == "sycl") && !cond {
			continue
		}
		out = append(out, o)
	}
	return out
}
