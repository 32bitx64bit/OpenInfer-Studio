// Package processes starts and stops external executables (llama-server)
// with an explicit path, working directory, env allowlist, and process-tree
// cleanup on every platform.
package processes

import (
	"fmt"
	"io"
	"os"
	"os/exec"
)

// Spec describes a managed child process.
type Spec struct {
	Exe     string            // absolute path to executable (never via PATH)
	Args    []string          // argument vector, never a shell string
	Dir     string            // explicit working directory
	Env     map[string]string // allowlisted environment variables
	LogName string
}

// Handle is a running managed process.
type Handle struct {
	Cmd *exec.Cmd
}

// envAllowlist is the only ambient variables inherited by children; anything
// in Spec.Env is added/overridden on top.
var envAllowlist = []string{
	"PATH", "SystemRoot", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR",
	"HOME", "USERPROFILE", "LANG", "LC_ALL", "LC_CTYPE",
	"LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
	"CUDA_PATH", "CUDA_HOME", "ROCM_PATH", "HIP_PATH",
	"VK_ICD_FILENAMES", "VK_DRIVER_FILES", "VK_ADD_LAYER_PATH",
	"DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR",
}

// buildEnv composes the child environment from the allowlist plus overrides.
func buildEnv(overrides map[string]string) []string {
	env := map[string]string{}
	for _, k := range envAllowlist {
		if v, ok := os.LookupEnv(k); ok {
			env[k] = v
		}
	}
	for k, v := range overrides {
		env[k] = v
	}
	out := make([]string, 0, len(env))
	for k, v := range env {
		out = append(out, k+"="+v)
	}
	return out
}

func prepareCmd(spec Spec) (*exec.Cmd, error) {
	if spec.Exe == "" {
		return nil, fmt.Errorf("empty executable path")
	}
	cmd := exec.Command(spec.Exe, spec.Args...)
	cmd.Dir = spec.Dir
	cmd.Env = buildEnv(spec.Env)
	if err := platformSetup(cmd); err != nil {
		return nil, err
	}
	return cmd, nil
}

// Start launches the process with platform-specific supervision (process
// groups + parent-death signal on Linux/macOS, Job Object on Windows).
func Start(spec Spec, stdout, stderr *os.File) (*Handle, error) {
	cmd, err := prepareCmd(spec)
	if err != nil {
		return nil, err
	}
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting %s: %w", spec.Exe, err)
	}
	h := &Handle{Cmd: cmd}
	if err := platformAfterStart(cmd); err != nil {
		// Supervision attach failure: kill the child rather than leave an
		// unsupervised process.
		h.KillTree()
		return nil, err
	}
	return h, nil
}

// StartPiped launches the process with stdin/stdout pipes for protocol
// drivers (e.g. DiffusionGemma visual server). stderr is written to errOut
// when non-nil.
func StartPiped(spec Spec, errOut io.Writer) (h *Handle, stdin io.WriteCloser, stdout io.ReadCloser, err error) {
	cmd, err := prepareCmd(spec)
	if err != nil {
		return nil, nil, nil, err
	}
	stdin, err = cmd.StdinPipe()
	if err != nil {
		return nil, nil, nil, err
	}
	stdout, err = cmd.StdoutPipe()
	if err != nil {
		stdin.Close()
		return nil, nil, nil, err
	}
	if errOut != nil {
		cmd.Stderr = errOut
	}
	if err := cmd.Start(); err != nil {
		stdin.Close()
		stdout.Close()
		return nil, nil, nil, fmt.Errorf("starting %s: %w", spec.Exe, err)
	}
	h = &Handle{Cmd: cmd}
	if err := platformAfterStart(cmd); err != nil {
		h.KillTree()
		stdin.Close()
		stdout.Close()
		return nil, nil, nil, err
	}
	return h, stdin, stdout, nil
}

// Signal requests graceful shutdown (SIGTERM on unix, CTRL_BREAK/Terminate on
// Windows via KillTree fallback).
func (h *Handle) Signal() error { return platformSignal(h.Cmd) }

// KillTree terminates the process and its entire child tree.
func (h *Handle) KillTree() error { return platformKillTree(h.Cmd) }

// Wait blocks until the process exits and returns the exit code.
func (h *Handle) Wait() (int, error) {
	err := h.Cmd.Wait()
	if err == nil {
		return 0, nil
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return ee.ExitCode(), err
	}
	return -1, err
}
