//go:build !windows

package orchestrate

import (
	"os/exec"
	"syscall"
	"time"
)

// configureProcGroup places the child in its own process group so the whole
// tree can be signalled on cancellation.
func configureProcGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// killProcTree SIGKILLs the child's entire process group, falling back to
// killing the direct child when the group is gone.
func killProcTree(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	pgid, err := syscall.Getpgid(cmd.Process.Pid)
	if err == nil && pgid > 0 {
		if kerr := syscall.Kill(-pgid, syscall.SIGKILL); kerr == nil || kerr == syscall.ESRCH {
			return nil
		}
	}
	_ = cmd.Process.Kill()
	// Give the reaper a moment so zombies do not accumulate.
	time.Sleep(10 * time.Millisecond)
	return nil
}
