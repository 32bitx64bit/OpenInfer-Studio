//go:build linux

package hardware

import (
	"bufio"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

func detectPlatform(i *Info) {
	detectCPU(i)
	detectMem(i)
	detectOSVersion(i)
	detectGPUs(i)
	detectAccel(i)
}

func detectOSVersion(i *Info) {
	b, err := os.ReadFile("/etc/os-release")
	if err != nil {
		i.Probes = append(i.Probes, Probe{Name: "os-release", OK: false, Detail: err.Error()})
		return
	}
	for _, line := range strings.Split(string(b), "\n") {
		if v, ok := strings.CutPrefix(line, "PRETTY_NAME="); ok {
			i.OSVersion = strings.Trim(v, `"`)
			i.Probes = append(i.Probes, Probe{Name: "os-release", OK: true})
			return
		}
	}
}

func detectCPU(i *Info) {
	f, err := os.Open("/proc/cpuinfo")
	if err != nil {
		i.Probes = append(i.Probes, Probe{Name: "cpuinfo", OK: false, Detail: err.Error()})
		return
	}
	defer f.Close()
	features := map[string]bool{}
	physical := map[string]bool{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		switch k {
		case "model name":
			if i.CPUModel == "" {
				i.CPUModel = v
			}
		case "flags", "Features":
			for _, fl := range strings.Fields(v) {
				switch strings.ToLower(fl) {
				case "avx", "avx2", "avx512f", "fma", "f16c", "sse4_2", "neon", "asimd":
					features[strings.ToLower(fl)] = true
				}
			}
		case "physical id", "core id":
			physical[k+"="+v] = true
		}
	}
	for fl := range features {
		i.CPUFeatures = append(i.CPUFeatures, fl)
	}
	// physical ids+core ids double-count; halve as an approximation.
	if n := len(physical) / 2; n > 0 {
		i.PhysicalCores = n
	} else {
		i.PhysicalCores = i.LogicalCores
	}
	i.Probes = append(i.Probes, Probe{Name: "cpuinfo", OK: true})
}

func detectMem(i *Info) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		i.Probes = append(i.Probes, Probe{Name: "meminfo", OK: false, Detail: err.Error()})
		return
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		fields := strings.Fields(v)
		if len(fields) == 0 {
			continue
		}
		kb, err := strconv.ParseUint(fields[0], 10, 64)
		if err != nil {
			continue
		}
		switch strings.TrimSpace(k) {
		case "MemTotal":
			i.RAMTotal = kb * 1024
		case "MemAvailable":
			i.RAMAvailable = kb * 1024
		}
	}
	i.Probes = append(i.Probes, Probe{Name: "meminfo", OK: true})
}

// pciVendor maps /sys/class/drm PCI vendor IDs.
var pciVendor = map[string]string{
	"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel",
}

// sysfsVRAMTotal reads dedicated VRAM from the DRM device directory
// (mem_info_vram_total, exposed by amdgpu and some others). Returns 0 when
// the driver does not report a dedicated pool — e.g. integrated GPUs, which
// share system RAM and are treated as unified memory.
func sysfsVRAMTotal(deviceDir string) uint64 {
	b, err := os.ReadFile(filepath.Join(deviceDir, "mem_info_vram_total"))
	if err != nil {
		return 0
	}
	n, err := strconv.ParseUint(strings.TrimSpace(string(b)), 10, 64)
	if err != nil {
		return 0
	}
	return n
}

func gpuVendorPresent(gpus []GPU, vendor string) bool {
	for _, g := range gpus {
		if g.Vendor == vendor {
			return true
		}
	}
	return false
}

func detectGPUs(i *Info) {
	entries, err := filepath.Glob("/sys/class/drm/card*/device/vendor")
	if err == nil {
		for _, vfile := range entries {
			b, err := os.ReadFile(vfile)
			if err != nil {
				continue
			}
			vendor := pciVendor[strings.TrimSpace(string(b))]
			if vendor == "" {
				vendor = "unknown"
			}
			i.GPUs = append(i.GPUs, GPU{
				Vendor: vendor,
				Name:   vendor + " GPU",
				VRAM:   sysfsVRAMTotal(filepath.Dir(vfile)),
			})
		}
	}
	// Enrich with nvidia-smi when available (names, VRAM, driver).
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits").Output()
	if err == nil {
		i.GPUs = i.GPUs[:0]
		for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			parts := strings.Split(line, ",")
			if len(parts) < 3 {
				continue
			}
			mib, _ := strconv.ParseUint(strings.TrimSpace(parts[1]), 10, 64)
			i.GPUs = append(i.GPUs, GPU{
				Name:   strings.TrimSpace(parts[0]),
				Vendor: "nvidia",
				VRAM:   mib << 20,
				Driver: strings.TrimSpace(parts[2]),
			})
		}
		i.CUDA = true
		i.Probes = append(i.Probes, Probe{Name: "nvidia-smi", OK: true})
	} else if gpuVendorPresent(i.GPUs, "nvidia") {
		i.Probes = append(i.Probes, Probe{Name: "nvidia-smi", OK: false,
			Detail: "NVIDIA device present but nvidia-smi failed; driver may be missing"})
	}
	i.Probes = append(i.Probes, Probe{Name: "drm", OK: len(i.GPUs) > 0})
}

func detectAccel(i *Info) {
	// Vulkan: loader library present.
	for _, p := range []string{"/usr/lib/libvulkan.so.1", "/usr/lib64/libvulkan.so.1",
		"/usr/lib/x86_64-linux-gnu/libvulkan.so.1", "/usr/lib/aarch64-linux-gnu/libvulkan.so.1"} {
		if _, err := os.Stat(p); err == nil {
			i.Vulkan = true
			break
		}
	}
	if !i.Vulkan {
		if _, err := exec.LookPath("vulkaninfo"); err == nil {
			i.Vulkan = true
		}
	}
	i.Probes = append(i.Probes, Probe{Name: "vulkan", OK: i.Vulkan})

	// ROCm/HIP.
	if _, err := os.Stat("/opt/rocm"); err == nil {
		i.HIP = true
	} else if _, err := exec.LookPath("rocminfo"); err == nil {
		i.HIP = true
	}
	i.Probes = append(i.Probes, Probe{Name: "rocm", OK: i.HIP})

	// CUDA fallback: toolkit dirs (nvidia-smi success already set i.CUDA).
	if !i.CUDA {
		if ok, _ := filepath.Glob("/usr/local/cuda*/bin/nvcc"); len(ok) > 0 {
			i.CUDA = true
		}
	}
	i.Probes = append(i.Probes, Probe{Name: "cuda", OK: i.CUDA})
	i.Probes = append(i.Probes, Probe{Name: "metal", OK: false, Detail: "not available on Linux"})
	// SYCL: presence of oneAPI runtime libraries.
	for _, p := range []string{"/usr/lib/libze_loader.so.1", "/opt/intel/oneapi"} {
		if _, err := os.Stat(p); err == nil {
			i.SYCL = true
			break
		}
	}
	i.Probes = append(i.Probes, Probe{Name: "sycl", OK: i.SYCL})
}

func diskFree(dir string) uint64 {
	var st syscall.Statfs_t
	// Stat the deepest existing ancestor.
	for dir != "" && dir != "/" {
		if err := syscall.Statfs(dir, &st); err == nil {
			return st.Bavail * uint64(st.Bsize)
		}
		dir = filepath.Dir(dir)
	}
	return 0
}
