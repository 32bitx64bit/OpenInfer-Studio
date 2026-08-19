package orchestrate

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// helperPath returns the test binary re-executed as a helper process.
// Helper mode is selected by the QUANTLAB_HELPER env var; argv[1:] is the
// helper's own arguments.
func helperPath() (string, []string) {
	return os.Args[0], []string{"-test.run=TestHelperProcess", "--"}
}

// TestHelperProcess is not a test; it is the subprocess body dispatched by
// QUANTLAB_HELPER.
func TestHelperProcess(t *testing.T) {
	mode := os.Getenv("QUANTLAB_HELPER")
	if mode == "" {
		return
	}
	// argv after "--"
	var args []string
	for i, a := range os.Args {
		if a == "--" {
			args = os.Args[i+1:]
			break
		}
	}
	switch mode {
	case "echo":
		fmt.Println(strings.Join(args, "\x1f"))
	case "flood":
		for i := 0; i < 4096; i++ {
			fmt.Println(strings.Repeat("x", 1024))
			fmt.Fprintln(os.Stderr, strings.Repeat("e", 1024))
		}
	case "exit7":
		fmt.Fprint(os.Stderr, "boom")
		os.Exit(7)
	case "tree":
		// Spawn a grandchild that sleeps, record its pid, then sleep.
		grand := exec.Command(os.Args[0], "-test.run=TestHelperProcess", "--")
		grand.Env = []string{"QUANTLAB_HELPER=sleeper"}
		if err := grand.Start(); err != nil {
			os.Exit(2)
		}
		if len(args) > 0 {
			os.WriteFile(args[0], []byte(strconv.Itoa(grand.Process.Pid)), 0o644)
		}
		fmt.Println("ready")
		time.Sleep(time.Hour)
	case "sleeper":
		time.Sleep(time.Hour)
	case "pulse":
		for i := 0; i < 4; i++ {
			fmt.Printf("tick %d\n", i)
			if i < 3 {
				time.Sleep(600 * time.Millisecond)
			}
		}
	case "help":
		fmt.Print(`usage: llama-quantize [options] model-f32.gguf [model-quant.gguf] type [nthreads]
  --imatrix file
  --tensor-type file
  --output-tensor-type type
  --token-embedding-type type
  --keep-split
  --pure
  --dry-run
  --version
allowed types: Q4_K_M, Q8_0, IQ2_XXS
version: 1.2.3
`)
	}
	os.Exit(0)
}

func runHelper(t *testing.T, r OSRunner, mode string, args ...string) (Result, error) {
	t.Helper()
	bin, pre := helperPath()
	iv := Invocation{
		Tool: ToolLlamaQuantize, // any valid tool; the path decides behavior
		Path: bin,
		Argv: append(append([]string{}, pre...), args...),
		Env:  []string{"QUANTLAB_HELPER=" + mode},
	}
	return r.Run(context.Background(), iv)
}

func TestOSRunnerMetacharsLiteral(t *testing.T) {
	r := OSRunner{}
	evil := []string{"a;rm -rf /", "$(touch /tmp/pwned)", "`id`", "x|y", "a&&b", "*", "line\nbreak", "  spaces  "}
	res, err := runHelper(t, r, "echo", evil...)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Split(strings.TrimRight(res.Stdout, "\n"), "\x1f")
	if len(got) != len(evil) {
		t.Fatalf("argc = %d, want %d (%q)", len(got), len(evil), res.Stdout)
	}
	for i := range evil {
		if got[i] != evil[i] {
			t.Errorf("argv[%d] = %q, want %q", i, got[i], evil[i])
		}
	}
	if _, err := os.Stat("/tmp/pwned"); err == nil {
		t.Fatal("shell metacharacters were interpreted")
	}
}

func TestOSRunnerEnvIsExplicit(t *testing.T) {
	r := OSRunner{} // nil env, invocation supplies its own
	res, err := runHelper(t, r, "echo", "hi")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(res.Stdout, "hi") {
		t.Fatalf("stdout = %q", res.Stdout)
	}
	// No env anywhere is refused.
	bin, pre := helperPath()
	iv := Invocation{Tool: ToolLlamaQuantize, Path: bin, Argv: pre}
	if _, err := r.Run(context.Background(), iv); err == nil {
		t.Fatal("runner accepted invocation with no explicit environment")
	}
}

func TestOSRunnerWorkDir(t *testing.T) {
	dir := t.TempDir()
	r := OSRunner{WorkDir: dir}
	res, err := runHelper(t, r, "echo", "ok")
	if err != nil {
		t.Fatal(err)
	}
	_ = res
	// Invocation-level WorkDir wins.
	r2 := OSRunner{WorkDir: "/nonexistent-dir-should-not-be-used"}
	res, err = r2.Run(context.Background(), Invocation{
		Tool: ToolLlamaQuantize, Path: os.Args[0],
		Argv:    []string{"-test.run=TestHelperProcess", "--"},
		Env:     []string{"QUANTLAB_HELPER=echo"},
		WorkDir: dir,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.ExitCode != 0 {
		t.Fatalf("exit = %d", res.ExitCode)
	}
}

func TestOSRunnerOutputLimits(t *testing.T) {
	r := OSRunner{MaxOutputBytes: 1 << 16}
	res, err := runHelper(t, r, "flood")
	if err != nil {
		t.Fatal(err)
	}
	if int64(len(res.Stdout)) > 1<<16 || int64(len(res.Stderr)) > 1<<16 {
		t.Fatalf("capture exceeded bound: %d/%d", len(res.Stdout), len(res.Stderr))
	}
	if !res.StdoutTruncated || !res.StderrTruncated {
		t.Fatal("truncation not reported")
	}
}

func TestOSRunnerNonzeroExit(t *testing.T) {
	r := OSRunner{}
	res, err := runHelper(t, r, "exit7")
	if err == nil {
		t.Fatal("nonzero exit not reported")
	}
	var ee *ExitError
	if !errors.As(err, &ee) {
		t.Fatalf("error is %T, want *ExitError", err)
	}
	if ee.ExitCode != 7 || res.ExitCode != 7 {
		t.Fatalf("exit code = %d/%d, want 7", ee.ExitCode, res.ExitCode)
	}
	if !strings.Contains(ee.StderrTail, "boom") {
		t.Fatalf("stderr tail = %q", ee.StderrTail)
	}
}

func TestOSRunnerIdleTimeout(t *testing.T) {
	start := time.Now()
	_, err := runHelper(t, OSRunner{IdleTimeout: 200 * time.Millisecond}, "sleeper")
	var idle *IdleTimeoutError
	if !errors.As(err, &idle) {
		t.Fatalf("err = %T %v, want *IdleTimeoutError", err, err)
	}
	if time.Since(start) > 2*time.Second {
		t.Fatal("idle process was not terminated promptly")
	}
}

func TestOSRunnerOutputResetsIdleTimeoutAndStreamsLines(t *testing.T) {
	var lines []string
	r := OSRunner{
		IdleTimeout: time.Second,
		OnOutput: func(event OutputEvent) {
			lines = append(lines, event.Line)
		},
	}
	if _, err := runHelper(t, r, "pulse"); err != nil {
		t.Fatal(err)
	}
	if len(lines) != 4 || lines[0] != "tick 0" || lines[3] != "tick 3" {
		t.Fatalf("streamed lines = %#v", lines)
	}
}

func TestOSRunnerTimeoutKillsTree(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("process-group kill test is unix-only")
	}
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
