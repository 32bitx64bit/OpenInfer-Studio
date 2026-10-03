//go:build !linux

package processes

// Other platforms report a crash through process exit.
func (h *Handle) IsCoreDumping() bool { return false }
