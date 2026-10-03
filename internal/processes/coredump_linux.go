//go:build linux

package processes

import (
	"os"
	"strconv"
	"strings"
)

// IsCoreDumping distinguishes a crashed process from a running one. Linux
// keeps a process alive while a piped core-dump handler copies its memory;
// Wait may not return for minutes even though inference has already stopped.
func (h *Handle) IsCoreDumping() bool {
	if h == nil || h.Cmd == nil || h.Cmd.Process == nil {
		return false
	}
	b, err := os.ReadFile("/proc/" + strconv.Itoa(h.Cmd.Process.Pid) + "/status")
	return err == nil && coreDumpingStatus(string(b))
}

func coreDumpingStatus(status string) bool {
	for _, line := range strings.Split(status, "\n") {
		key, value, ok := strings.Cut(line, ":")
		if ok && key == "CoreDumping" {
			return strings.TrimSpace(value) == "1"
		}
	}
	return false
}
