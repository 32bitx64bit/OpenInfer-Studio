package runtimes

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// ToolInfo describes one sibling binary next to llama-server.
type ToolInfo struct {
	Name    string   `json:"name"`
	Path    string   `json:"path,omitempty"`
	Present bool     `json:"present"`
	Error   string   `json:"error,omitempty"`
	Flags   []string `json:"flags,omitempty"`
}

// ToolsSnapshot is the persisted inventory of llama.cpp sibling tools.
type ToolsSnapshot struct {
	Quantize   ToolInfo `json:"quantize"`
	IMatrix    ToolInfo `json:"imatrix"`
	Perplexity ToolInfo `json:"perplexity"`
	GGUFSplit  ToolInfo `json:"gguf_split"`
}

var toolFlagRe = regexp.MustCompile(`(?:^|[\s,\[|])(-{1,2}[a-zA-Z][a-zA-Z0-9\-]*)`)
var quantizeSizeRe = regexp.MustCompile(`(?i)quant size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(bytes?|[kmgt]i?b)`)

// Tools returns (and caches on disk as tools.json) sibling-tool info for a runtime.
func (m *Manager) Tools(id string) (ToolsSnapshot, error) {
	r, err := m.Get(id)
	if err != nil {
		return ToolsSnapshot{}, err
	}
	cache := filepath.Join(r.InstallDir, "tools.json")
	snap := probeTools(r.ExecutablePath)
	if b, err := json.MarshalIndent(snap, "", "  "); err == nil {
		_ = os.WriteFile(cache, b, 0o644)
	}
	return snap, nil
}

func probeTools(serverExe string) ToolsSnapshot {
	return ToolsSnapshot{
		Quantize:   probeOne(serverExe, "llama-quantize"),
		IMatrix:    probeOne(serverExe, "llama-imatrix"),
		Perplexity: probeOne(serverExe, "llama-perplexity"),
		GGUFSplit:  probeOne(serverExe, "llama-gguf-split"),
	}
}

func probeOne(serverExe, name string) ToolInfo {
	info := ToolInfo{Name: name}
	p, err := FindSibling(serverExe, name)
	if err != nil {
		info.Error = err.Error()
		return info
	}
	info.Present = true
	info.Path = p
	help, herr := ProbeHelp(p)
	if herr != nil {
		info.Error = herr.Error()
		return info
	}
	info.Flags = parseToolFlags(help)
	return info
}

func parseToolFlags(help string) []string {
	seen := map[string]bool{}
	var out []string
	for _, m := range toolFlagRe.FindAllStringSubmatch(help, -1) {
		flag := m[1]
		if flagRemoved(help, flag) {
			continue
		}
		if seen[flag] {
			continue
		}
		seen[flag] = true
		out = append(out, flag)
	}
	return out
}

// ToolHasFlag reports whether a probed tool advertised flag (long or short).
func ToolHasFlag(flags []string, want string) bool {
	want = strings.TrimSpace(want)
	for _, f := range flags {
		if f == want {
			return true
		}
	}
	return false
}

// RunTool captures a probed sibling tool's output using the same runtime
// library path setup as the server probes.
func RunTool(ctx context.Context, tool ToolInfo, args []string) (string, error) {
	if !tool.Present || tool.Path == "" {
		return "", fmt.Errorf("%s is not available", tool.Name)
	}
	cmd := exec.CommandContext(ctx, tool.Path, args...)
	cmd.Dir = filepath.Dir(tool.Path)
	cmd.Env = append(os.Environ(), LibPathEnv(tool.Path)...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("%s: %w", tool.Name, err)
	}
	return string(out), nil
}

// ParseQuantizeSize extracts llama-quantize's final dry-run size. New and
// patched builds may report bytes; released builds commonly report MiB.
func ParseQuantizeSize(output string) (int64, bool) {
	matches := quantizeSizeRe.FindAllStringSubmatch(output, -1)
	if len(matches) == 0 {
		return 0, false
	}
	m := matches[len(matches)-1]
	if len(m) != 3 {
		return 0, false
	}
	n, err := strconv.ParseFloat(m[1], 64)
	if err != nil || n < 0 {
		return 0, false
	}
	switch strings.ToLower(m[2]) {
	case "byte", "bytes":
	case "kb":
		n *= 1e3
	case "kib":
		n *= 1 << 10
	case "mb":
		n *= 1e6
	case "mib":
		n *= 1 << 20
	case "gb":
		n *= 1e9
	case "gib":
		n *= 1 << 30
	case "tb":
		n *= 1e12
	case "tib":
		n *= 1 << 40
	default:
		return 0, false
	}
	if n > math.MaxInt64 {
		return 0, false
	}
	return int64(math.Round(n)), true
}
