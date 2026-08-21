package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeTinyGGUF(t *testing.T, path string) {
	t.Helper()
	type tt struct {
		name  string
		dims  []uint64
		ggml  uint32
		float bool
	}
	tensors := []tt{
		{"blk.0.attn_q.weight", []uint64{256, 256}, 1, false}, // F16
		{"blk.0.attn_norm.weight", []uint64{256}, 0, true},    // F32
	}
	align := func(n, a uint64) uint64 { return (n + a - 1) / a * a }
	var rel []uint64
	var cur uint64
	lengths := make([]uint64, len(tensors))
	for i, ts := range tensors {
		elems := uint64(1)
		for _, d := range ts.dims {
			elems *= d
		}
		var l uint64
		if ts.float {
			l = elems * 4
		} else {
			l = elems * 2
		}
		lengths[i] = l
		cur = align(cur, 32)
		rel = append(rel, cur)
		cur += l
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var hdr [24]byte
	binary.LittleEndian.PutUint32(hdr[0:4], 0x46554747)
	binary.LittleEndian.PutUint32(hdr[4:8], 3)
	binary.LittleEndian.PutUint64(hdr[8:16], uint64(len(tensors)))
	binary.LittleEndian.PutUint64(hdr[16:24], 2)
	f.Write(hdr[:])
	kv := func(k, v string) {
		var b [8]byte
		var n [4]byte
		binary.LittleEndian.PutUint64(b[:], uint64(len(k)))
		f.Write(b[:])
		f.WriteString(k)
		binary.LittleEndian.PutUint32(n[:], 8)
		f.Write(n[:])
		binary.LittleEndian.PutUint64(b[:], uint64(len(v)))
		f.Write(b[:])
		f.WriteString(v)
	}
	kv("general.architecture", "llama")
	kv("general.name", "tiny")
	for i, ts := range tensors {
		var b [8]byte
		var n [4]byte
		binary.LittleEndian.PutUint64(b[:], uint64(len(ts.name)))
		f.Write(b[:])
		f.WriteString(ts.name)
		binary.LittleEndian.PutUint32(n[:], uint32(len(ts.dims)))
		f.Write(n[:])
		for _, d := range ts.dims {
			binary.LittleEndian.PutUint64(b[:], d)
			f.Write(b[:])
		}
		binary.LittleEndian.PutUint32(n[:], ts.ggml)
		f.Write(n[:])
		binary.LittleEndian.PutUint64(b[:], rel[i])
		f.Write(b[:])
	}
	pos, _ := f.Seek(0, 1)
	dataStart := align(uint64(pos), 32)
	if pad := int64(dataStart) - pos; pad > 0 {
		f.Write(make([]byte, pad))
	}
	for i := range tensors {
		if p, _ := f.Seek(0, 1); p < int64(dataStart+rel[i]) {
			f.Write(make([]byte, int64(dataStart+rel[i])-p))
		}
		f.Write(make([]byte, lengths[i]))
	}
}

func TestCLIErrorPaths(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want string
	}{
		{"no args", nil, "missing command"},
		{"unknown", []string{"frobnicate"}, "unknown command"},
		{"plan missing src", []string{"plan", "-state-dir", "/tmp/x"}, "-src"},
		{"plan missing tools", []string{"plan", "-src", "/tmp/x.gguf", "-state-dir", "/tmp/x", "-calibration-dir", "/tmp/x", "-budget-bytes", "1000"}, "-quantize"},
		{"resume missing flags", []string{"resume"}, "-state-dir"},
		{"status missing flags", []string{"status", "-state-dir", "/tmp/x"}, "-run"},
		{"resume unknown run", []string{"resume", "-state-dir", "/tmp/quantlab-missing", "-run", "nope"}, "load"},
		{"status unknown run", []string{"status", "-state-dir", "/tmp/quantlab-missing", "-run", "nope"}, "load"},
		{"plan bad effort", []string{"plan", "-effort", "ludicrous"}, "unknown effort"},
	}
	for _, tc := range cases {
		var buf bytes.Buffer
		err := run(tc.args, &buf)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: got %v, want containing %q", tc.name, err, tc.want)
		}
	}
}

func TestCLIPlanBadGGUF(t *testing.T) {
	dir := t.TempDir()
	junk := filepath.Join(dir, "junk.gguf")
	if err := os.WriteFile(junk, []byte("junk"), 0o644); err != nil {
		t.Fatal(err)
	}
	calib := filepath.Join(dir, "corpus")
	os.MkdirAll(calib, 0o755)
	os.WriteFile(filepath.Join(calib, "a.txt"), []byte("hello world hello"), 0o644)
	var buf bytes.Buffer
	err := run([]string{"plan",
		"-src", junk,
		"-state-dir", filepath.Join(dir, "state"),
		"-calibration-dir", calib,
		"-quantize", junk,
		"-perplexity", junk,
		"-budget-bytes", "100000",
		"-run", "bad",
	}, &buf)
	if err == nil || !strings.Contains(err.Error(), "GGUF") {
		t.Fatalf("got %v, want GGUF error", err)
	}
}

func TestCLIPlanGatesNone(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "tiny.gguf")
	writeTinyGGUF(t, src)
	calib := filepath.Join(dir, "corpus")
	os.MkdirAll(calib, 0o755)
	os.WriteFile(filepath.Join(calib, "a.txt"), []byte("hello world hello world"), 0o644)
	stateDir := filepath.Join(dir, "state")
	var buf bytes.Buffer
	if err := run([]string{"plan",
		"-src", src,
		"-state-dir", stateDir,
		"-calibration-dir", calib,
		"-quantize", src,
		"-perplexity", src,
		"-budget-bytes", "100000",
		"-gates", "none",
		"-run", "optout",
	}, &buf); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(stateDir, "optout.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"gatesOptOut": true`) {
		t.Fatalf("opt-out not persisted:\n%s", data)
	}
}

func TestCLIPlanStatusResumeSmoke(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "tiny.gguf")
	writeTinyGGUF(t, src)
	calib := filepath.Join(dir, "corpus")
	if err := os.MkdirAll(calib, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(calib, "a.txt"), []byte("hello world hello world"), 0o644); err != nil {
		t.Fatal(err)
	}
	stateDir := filepath.Join(dir, "state")
	runID := "smoke" + time.Now().Format("150405")

	var buf bytes.Buffer
	if err := run([]string{"plan",
		"-src", src,
		"-state-dir", stateDir,
		"-calibration-dir", calib,
		"-out", filepath.Join(dir, "out"),
		"-quantize", src,
		"-perplexity", src,
		"-budget-bytes", "100000",
		"-threads", "2", "-ctx", "512",
		"-gates", "mean-kld=0.5",
		"-effort", "fast",
		"-run", runID,
	}, &buf); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "planned") || !strings.Contains(buf.String(), "effort=fast") {
		t.Fatalf("plan output: %q", buf.String())
	}

	buf.Reset()
	if err := run([]string{"status", "-state-dir", stateDir, "-run", runID}, &buf); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	for _, want := range []string{"run " + runID, "next stage: assemble", "[ ] assemble", "budget 100000"} {
		if !strings.Contains(out, want) {
			t.Errorf("status missing %q in:\n%s", want, out)
		}
	}

	// Resume executes exactly one stage (assemble needs no external tools).
	buf.Reset()
	if err := run([]string{"resume", "-state-dir", stateDir, "-run", runID, "-stage-limit", "1"}, &buf); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "stage assemble") {
		t.Fatalf("resume output: %q", buf.String())
	}

	buf.Reset()
	if err := run([]string{"status", "-state-dir", stateDir, "-run", runID}, &buf); err != nil {
		t.Fatal(err)
	}
	out = buf.String()
	if !strings.Contains(out, "next stage: anchor") || !strings.Contains(out, "bank: 2 tensors") {
		t.Errorf("post-resume status:\n%s", out)
	}
}
