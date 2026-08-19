//go:build windows

package orchestrate

import (
	"os/exec"
	"syscall"
)

// configureProcGroup starts the child with CREATE_NEW_PROCESS_GROUP so it
// can be addressed independently of the parent console group. Full Job
// Object tree cleanup requires x/sys; with the standard library alone the
// best available cleanup is killing the direct child on cancellation.
func configureProcGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP}
}

// killProcTree kills the direct child. Descendant cleanup beyond this needs
// Job Objects (golang.org/x/sys/windows), which this module avoids.
func killProcTree(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	return cmd.Process.Kill()
}
