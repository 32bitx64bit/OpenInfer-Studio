//go:build unix

package orchestrate

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestOSRunnerTimeoutKillsTree(t *testing.T) {
	dir := t.TempDir()
	pidFile := filepath.Join(dir, "grand.pid")
	bin, pre := helperPath()
	ctx, cancel := context.WithTimeout(context.Background(), 400*time.Millisecond)
	defer cancel()
	r := OSRunner{}
	start := time.Now()
	_, err := r.Run(ctx, Invocation{
		Tool: ToolLlamaQuantize, Path: bin,
		Argv: append(append([]string{}, pre...), pidFile),
		Env:  []string{"QUANTLAB_HELPER=tree"},
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want deadline exceeded", err)
	}
	if time.Since(start) > 10*time.Second {
		t.Fatal("cancelled run did not return promptly")
	}
	data, err := os.ReadFile(pidFile)
	if err != nil {
		t.Fatalf("grandchild pid file: %v", err)
	}
	pid, _ := strconv.Atoi(strings.TrimSpace(string(data)))
	if pid <= 0 {
		t.Fatalf("bad grandchild pid %q", data)
	}
	// SIGKILL delivery is synchronous for our purposes; poll briefly.
	deadline := time.Now().Add(2 * time.Second)
	for {
		err := syscall.Kill(pid, 0)
		if err == syscall.ESRCH {
			return // grandchild reaped/killed: tree cleanup worked
		}
		if err != nil && err != syscall.EPERM {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("grandchild pid %d survived process-group kill", pid)
		}
		time.Sleep(20 * time.Millisecond)
	}
}
