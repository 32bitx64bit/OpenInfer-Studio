package runtimes

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
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

// ToolsSnapshot is the persisted inventory of quantize/imatrix/split tools.
type ToolsSnapshot struct {
	Quantize  ToolInfo `json:"quantize"`
	IMatrix   ToolInfo `json:"imatrix"`
	GGUFSplit ToolInfo `json:"gguf_split"`
}

var toolFlagRe = regexp.MustCompile(`(?:^|[\s,\[|])(-{1,2}[a-zA-Z][a-zA-Z0-9\-]*)`)

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
		Quantize:  probeOne(serverExe, "llama-quantize"),
		IMatrix:   probeOne(serverExe, "llama-imatrix"),
		GGUFSplit: probeOne(serverExe, "llama-gguf-split"),
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
