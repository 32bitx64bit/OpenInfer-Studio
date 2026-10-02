package mediagen

import (
	"fmt"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/runtimes"
)

// RuntimeCapabilities returns the stable-diffusion.cpp runtime a graph would
// run on (runtimeID, or the preferred sd.cpp runtime when empty) and the
// capability ids its sd-server --help advertises. sd builds' recorded
// capability snapshot comes from the llama.cpp parser and is nearly empty, so
// the help text is parsed here, probing the binary once when no snapshot
// exists. Results are cached per runtime: planning calls this on every edit.
func (m *Manager) RuntimeCapabilities(runtimeID string) (string, []string, error) {
	if m.rt == nil {
		return "", nil, fmt.Errorf("no runtime manager")
	}
	rt, err := m.ResolveRuntime("", runtimeID)
	if err != nil {
		return "", nil, err
	}
	m.mu.Lock()
	if caps, ok := m.capCache[rt.ID]; ok {
		m.mu.Unlock()
		return rt.ID, append([]string(nil), caps...), nil
	}
	m.mu.Unlock()

	help, _ := m.rt.HelpOutput(rt.ID)
	if strings.TrimSpace(help) == "" {
		exe, err := SDServerPath(rt.ExecutablePath)
		if err != nil {
			return rt.ID, nil, err
		}
		if help, err = runtimes.ProbeHelp(exe); err != nil {
			return rt.ID, nil, fmt.Errorf("could not read sd-server --help: %w", err)
		}
	}
	caps := ParseSDCapabilities(help)
	m.mu.Lock()
	m.capCache[rt.ID] = caps
	m.mu.Unlock()
	return rt.ID, append([]string(nil), caps...), nil
}
