package orchestrate

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"quantlab/core"
)

// fakeRunner is an in-memory Runner for planner tests; no real binaries run.
type fakeRunner struct {
	results []Result
	errs    []error
	argvs   [][]string
}

func (f *fakeRunner) Run(ctx context.Context, iv Invocation) (Result, error) {
	f.argvs = append(f.argvs, iv.Argv)
	if len(f.results) == 0 {
		return Result{}, errors.New("fake runner exhausted")
	}
	res := f.results[0]
	err := f.errs[0]
	f.results = f.results[1:]
	f.errs = f.errs[1:]
	return res, err
}

const quantizeHelp = `usage: llama-quantize [options] model-f32.gguf [model-quant.gguf] type [nthreads]
  --imatrix file                    use importance matrix
  --include-suffix pattern
  --exclude-suffix pattern
  --output-tensor-type type
  --token-embedding-type type
  --tensor-type file
  --keep-split
  --pure
  --dry-run
  --version
  --help
allowed quantization types: Q2_K Q3_K Q3_K_S Q3_K_M Q3_K_L Q4_0 Q4_1 Q4_K Q4_K_S Q4_K_M Q5_0 Q5_1 Q5_K Q5_K_S Q5_K_M Q6_K Q8_0 IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ2_M IQ3_XXS IQ3_XS IQ3_S IQ4_NL IQ4_XS F16 BF16
version: b4123
`

const oldQuantizeHelp = `usage: llama-quantize [--imatrix file] [--keep-split] model-f32.gguf model-quant.gguf type [nthreads]
version: b1000
`

func TestParseHelpFlagsAndVersion(t *testing.T) {
	caps := ParseHelp(ToolLlamaQuantize, "/bin/llama-quantize", quantizeHelp)
	for _, f := range []string{"--imatrix", "--tensor-type", "--output-tensor-type", "--token-embedding-type", "--keep-split", "--pure", "--dry-run", "--version"} {
		if !caps.Has(f) {
			t.Errorf("missing advertised flag %s", f)
		}
	}
	if caps.Has("--definitely-not-a-flag") {
		t.Error("phantom flag advertised")
	}
	if caps.Version != "b4123" {
		t.Errorf("version = %q", caps.Version)
	}
	found := false
	for _, ty := range caps.Types {
		if ty == "Q4_K_M" {
			found = true
		}
	}
	if !found {
		t.Errorf("types = %v, want Q4_K_M present", caps.Types)
	}
}

func TestProbeCapabilitiesViaRunner(t *testing.T) {
	fr := &fakeRunner{
		results: []Result{{Stdout: quantizeHelp}},
		errs:    []error{nil},
	}
	caps, err := ProbeCapabilities(context.Background(), fr, ToolLlamaQuantize, "/bin/llama-quantize")
	if err != nil {
		t.Fatal(err)
	}
	if !caps.Has("--imatrix") || caps.Version != "b4123" {
		t.Fatalf("caps = %+v", caps)
	}
	if len(fr.argvs) != 1 || fr.argvs[0][0] != "--help" {
		t.Fatalf("probe argv = %v", fr.argvs)
	}
	// Empty help is an error.
	fr2 := &fakeRunner{results: []Result{{}}, errs: []error{nil}}
	if _, err := ProbeCapabilities(context.Background(), fr2, ToolLlamaQuantize, "/bin/x"); err == nil {
		t.Fatal("empty help accepted")
	}
}

func fullQuantCaps() *Capabilities {
	c := ParseHelp(ToolLlamaQuantize, "/bin/llama-quantize", quantizeHelp)
	return &c
}

func TestPlanQuantizeArgvOrder(t *testing.T) {
	req := QuantizeRequest{
		ProfileID:      "p1",
		SourcePath:     "/src/f16.gguf",
		OutputPath:     "/out/q.gguf",
		Type:           core.DTypeIQ4_XS,
		ImatrixPath:    "/cal/imatrix.bin",
		TensorTypeFile: "/work/types.txt",
		OutputType:     core.DTypeF16,
		EmbeddingType:  core.DTypeF32,
		Pure:           true,
		KeepSplit:      true,
		DryRun:         true,
		Threads:        12,
	}
	iv, err := PlanQuantize(req, fullQuantCaps(), "/bin/llama-quantize")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"--imatrix", "/cal/imatrix.bin",
		"--tensor-type", "/work/types.txt",
		"--output-tensor-type", "F16",
		"--token-embedding-type", "F32",
		"--pure", "--keep-split", "--dry-run",
		"/src/f16.gguf", "/out/q.gguf", "IQ4_XS", "12",
	}
	if strings.Join(iv.Argv, "\x1f") != strings.Join(want, "\x1f") {
		t.Fatalf("argv = %v, want %v", iv.Argv, want)
	}
	if iv.Tool != ToolLlamaQuantize || iv.Path != "/bin/llama-quantize" {
		t.Fatalf("iv = %+v", iv)
	}

	// Minimal request: flags absent, no positional threads.
	min := QuantizeRequest{ProfileID: "p", SourcePath: "/a.gguf", OutputPath: "/b.gguf", Type: core.DTypeQ4_K_M}
	iv, err = PlanQuantize(min, fullQuantCaps(), "/bin/llama-quantize")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(iv.Argv, " ") != "/a.gguf /b.gguf Q4_K_M" {
		t.Fatalf("minimal argv = %v", iv.Argv)
	}
}

func TestPlanQuantizeCapabilityGating(t *testing.T) {
	oldCaps := ParseHelp(ToolLlamaQuantize, "/bin/old", oldQuantizeHelp)
	oldCapsP := &oldCaps
	req := QuantizeRequest{
		ProfileID: "p", SourcePath: "/a.gguf", OutputPath: "/b.gguf",
		Type: core.DTypeQ4_K_M, Pure: true, // --pure not advertised by old binary
	}
	if _, err := PlanQuantize(req, oldCapsP, "/bin/old"); err == nil {
		t.Fatal("unadvertised --pure accepted")
	} else if !strings.Contains(err.Error(), "--pure") {
		t.Fatalf("err = %v", err)
	}
	req.Pure = false
	req.TensorTypeFile = "/work/types.txt" // also unadvertised
	if _, err := PlanQuantize(req, oldCapsP, "/bin/old"); err == nil {
		t.Fatal("unadvertised --tensor-type accepted")
	}
	// Advertised subset passes.
	req.TensorTypeFile = ""
	req.Threads = 8
	req.ImatrixPath = "/im.bin" // --imatrix IS advertised in old help
	if _, err := PlanQuantize(req, oldCapsP, "/bin/old"); err != nil {
		t.Fatalf("advertised flags rejected: %v", err)
	}
	// Wrong-tool capabilities rejected.
	pplCaps := ParseHelp(ToolPerplexity, "/bin/p", "-m -f -c --kl-divergence-base")
	if _, err := PlanQuantize(req, &pplCaps, "/bin/old"); err == nil {
		t.Fatal("perplexity caps accepted for quantize")
	}
}

func TestPlanQuantizeTypeGating(t *testing.T) {
	// A caps-advertised type list gates the positional type argument
	// (fail-closed): a dtype the binary does not advertise is refused even
	// when every flag passes.
	narrow := ParseHelp(ToolLlamaQuantize, "/bin/narrow", `usage: llama-quantize model type
allowed quantization types: Q8_0 Q4_0
version: b5000
`)
	req := QuantizeRequest{
		ProfileID: "p", SourcePath: "/a.gguf", OutputPath: "/b.gguf",
		Type: core.DTypeQ4_K_M,
	}
	if _, err := PlanQuantize(req, &narrow, "/bin/narrow"); err == nil {
		t.Fatal("unadvertised quant type accepted")
	} else if !strings.Contains(err.Error(), "Q4_K_M") {
		t.Fatalf("err = %v", err)
	}
	req.Type = core.DTypeQ8_0
	if _, err := PlanQuantize(req, &narrow, "/bin/narrow"); err != nil {
		t.Fatalf("advertised type rejected: %v", err)
	}
	// An empty advertised list (older tools) disables type gating.
	legacy := ParseHelp(ToolLlamaQuantize, "/bin/legacy", oldQuantizeHelp)
	if len(legacy.Types) != 0 {
		t.Fatalf("legacy caps parsed types from untyped help: %v", legacy.Types)
	}
	req.Type = core.DTypeQ3_K_L
	if _, err := PlanQuantize(req, &legacy, "/bin/legacy"); err != nil {
		t.Fatalf("empty type list gated the request: %v", err)
	}
	// Nil caps permits everything (callers that never probed).
	if _, err := PlanQuantize(req, nil, "/bin/lq"); err != nil {
		t.Fatalf("nil caps rejected request: %v", err)
	}
}

func TestPlanQuantizePureIQ2SUsesIQ2M(t *testing.T) {
	req := QuantizeRequest{
		ProfileID: "p", SourcePath: "/a.gguf", OutputPath: "/b.gguf",
		Type: core.DTypeIQ2_S, ImatrixPath: "/im.bin", Pure: true,
	}
	iv, err := PlanQuantize(req, fullQuantCaps(), "/bin/lq")
	if err != nil {
		t.Fatal(err)
	}
	if got := iv.Argv[len(iv.Argv)-1]; got != "IQ2_M" {
		t.Fatalf("--pure IQ2_S argv type = %q, want IQ2_M; argv=%v", got, iv.Argv)
	}

	req.Pure = false
	iv, err = PlanQuantize(req, fullQuantCaps(), "/bin/lq")
	if err != nil {
		t.Fatal(err)
	}
	if got := iv.Argv[len(iv.Argv)-1]; got != "IQ2_S" {
		t.Fatalf("without --pure argv type = %q, want IQ2_S; argv=%v", got, iv.Argv)
	}

	noM := ParseHelp(ToolLlamaQuantize, "/bin/old", `usage: llama-quantize [options] model type
  --pure
  --imatrix file
allowed quantization types: Q8_0 IQ2_S IQ2_XS
version: b10000
`)
	req.Pure = true
	iv, err = PlanQuantize(req, &noM, "/bin/old")
	if err != nil {
		t.Fatal(err)
	}
	if got := iv.Argv[len(iv.Argv)-1]; got != "IQ2_S" {
		t.Fatalf("IQ2_M missing from help: argv type = %q, want IQ2_S; argv=%v", got, iv.Argv)
	}

	iv, err = PlanQuantize(req, nil, "/bin/lq")
	if err != nil {
		t.Fatal(err)
	}
	if got := iv.Argv[len(iv.Argv)-1]; got != "IQ2_M" {
		t.Fatalf("nil caps --pure IQ2_S argv type = %q, want IQ2_M; argv=%v", got, iv.Argv)
	}
}

func TestPlanQuantizeRequantizeRefusal(t *testing.T) {
	req := QuantizeRequest{
		ProfileID: "p", SourcePath: "/q4.gguf", OutputPath: "/q3.gguf",
		Type: core.DTypeQ3_K, SourceQuantized: true,
	}
	if _, err := PlanQuantize(req, fullQuantCaps(), "/bin/lq"); err == nil {
		t.Fatal("requantization accepted by default")
	}
	req.AllowRequantize = true
	if _, err := PlanQuantize(req, fullQuantCaps(), "/bin/lq"); err != nil {
		t.Fatalf("explicit override refused: %v", err)
	}
	// IQ type without imatrix is refused even with override.
	req.Type = core.DTypeIQ2_XXS
	if _, err := PlanQuantize(req, fullQuantCaps(), "/bin/lq"); err == nil {
		t.Fatal("IQ type without imatrix accepted")
	}
	req.ImatrixPath = "/im.bin"
	if _, err := PlanQuantize(req, fullQuantCaps(), "/bin/lq"); err != nil {
		t.Fatal(err)
	}
	// Non-float output type refused.
	req2 := QuantizeRequest{
		ProfileID: "p", SourcePath: "/a.gguf", OutputPath: "/b.gguf",
		Type: core.DTypeQ4_K_M, OutputType: core.DTypeQ6_K,
	}
	if _, err := PlanQuantize(req2, fullQuantCaps(), "/bin/lq"); err == nil {
		t.Fatal("quant output-tensor-type accepted")
	}
}

func TestSourceIsQuantized(t *testing.T) {
	bank := &core.TensorBank{SourcePath: "/m.gguf", Tensors: []core.TensorDesc{
		{Name: "a", DType: core.DTypeF16, Shape: []uint64{2, 2}, Elements: 4, Length: 8},
	}}
	if SourceIsQuantized(bank) {
		t.Fatal("float bank reported quantized")
	}
	bank.Tensors = append(bank.Tensors, core.TensorDesc{
		Name: "b", DType: core.DTypeQ6_K, Shape: []uint64{2, 256}, Elements: 512, Length: 420,
	})
	if !SourceIsQuantized(bank) {
		t.Fatal("quant bank reported unquantized")
	}
	if SourceIsQuantized(nil) {
		t.Fatal("nil bank reported quantized")
	}
}

func TestTensorTypeFileDeterministicSorted(t *testing.T) {
	opts := []core.TensorOption{
		{TensorName: "blk.9.ffn", Target: core.DTypeQ3_K, Bytes: 1},
		{TensorName: "blk.1.attn", Target: core.DTypeQ6_K, Bytes: 1},
		{TensorName: "blk.10.attn", Target: core.DTypeQ4_K_T, Bytes: 1},
		{TensorName: "output.weight", Target: core.DTypeQ6_K, Bytes: 1},
	}
	var buf bytes.Buffer
	if err := WriteTensorTypeFile(&buf, opts); err != nil {
		t.Fatal(err)
	}
	got := buf.String()
	want := "blk.1.attn Q6_K\nblk.10.attn Q4_K\nblk.9.ffn Q3_K\noutput.weight Q6_K\n"
	if got != want {
		t.Fatalf("type file = %q, want %q", got, want)
	}
	// Determinism: shuffled input yields identical bytes and digest.
	opts[0], opts[3] = opts[3], opts[0]
	opts[1], opts[2] = opts[2], opts[1]
	var buf2 bytes.Buffer
	if err := WriteTensorTypeFile(&buf2, opts); err != nil {
		t.Fatal(err)
	}
	if buf2.String() != got {
		t.Fatal("type file not deterministic under input permutation")
	}
	h1, _ := TensorTypeFileSHA256(opts)
	h2, _ := TensorTypeFileSHA256(reverseOpts(opts))
	if h1 != h2 || len(h1) != 64 {
		t.Fatalf("digests differ or malformed: %s vs %s", h1, h2)
	}
	// Duplicates refused.
	dup := append(append([]core.TensorOption{}, opts...), core.TensorOption{TensorName: "blk.9.ffn", Target: core.DTypeQ8_0, Bytes: 1})
	if _, err := TensorTypeFileBytes(dup); err == nil {
		t.Fatal("duplicate tensor accepted")
	}
}

func reverseOpts(o []core.TensorOption) []core.TensorOption {
	out := make([]core.TensorOption, len(o))
	for i, v := range o {
		out[len(o)-1-i] = v
	}
	return out
}

// TestTensorTypeFileRejectsInjection proves adversarial tensor names (as
// could arrive from a crafted GGUF header) cannot inject override lines or
// extra arguments into the --tensor-type file, even if every earlier
// validation layer was bypassed.
func TestTensorTypeFileRejectsInjection(t *testing.T) {
	bad := []string{
		"blk.0.weight\nfake.tensor Q8_0", // newline: injects a second override line
		"blk.0.weight\rQ8_0",             // carriage return
		" blk.0.weight",                  // leading space: column shift
		"blk.0.weight ",                  // trailing space
		"blk.0 weight",                   // embedded space: splits name/type
		"blk.0\tweight",
		"blk.0\x07weight",
		"blk.0 weight", // unicode line separator
		"blk.0weight", // next-line control
		"",
		strings.Repeat("n", core.MaxTensorNameLen+1),
	}
	for _, name := range bad {
		opts := []core.TensorOption{{TensorName: name, Target: core.DTypeQ8_0, Bytes: 64}}
		if _, err := TensorTypeFileBytes(opts); err == nil {
			t.Errorf("name %q accepted into tensor-type file", name)
		}
		var sb strings.Builder
		if err := WriteTensorTypeFile(&sb, opts); err == nil {
			t.Errorf("name %q written to tensor-type file", name)
		}
	}
	// A valid file holds exactly one "name TYPE" line per option.
	opts := []core.TensorOption{{TensorName: "blk.0.attn_q.weight", Target: core.DTypeQ6_K, Bytes: 64}}
	b, err := TensorTypeFileBytes(opts)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(b), "blk.0.attn_q.weight Q6_K\n"; got != want {
		t.Fatalf("file = %q, want %q", got, want)
	}
}

func TestAnchorBatchBuildRecordCleanup(t *testing.T) {
	dir := t.TempDir()
	anchors := []core.Anchor{
		{Kind: core.AnchorEmbedding, Pattern: "*output*", MinDType: core.DTypeQ8_0},
		{Kind: core.AnchorAttention, Pattern: "*attn*", MinDType: core.DTypeQ6_K},
		{Kind: core.AnchorExplicit, Name: "blk.0.ffn", MinDType: core.DTypeF16}, // float: skipped
	}
	b, err := BuildAnchorBatch("/src.gguf", anchors, dir, "prof1")
	if err != nil {
		t.Fatal(err)
	}
	if len(b.Jobs) != 2 {
		t.Fatalf("jobs = %d, want 2", len(b.Jobs))
	}
	// Sorted by floor then tensor.
	if b.Jobs[0].Floor != core.DTypeQ6_K || b.Jobs[1].Floor != core.DTypeQ8_0 {
		t.Fatalf("job order = %+v", b.Jobs)
	}
	// Deterministic output names, distinct, inside workdir.
	seen := map[string]bool{}
	for i, j := range b.Jobs {
		if !strings.HasPrefix(filepath.Base(j.Request.OutputPath), "anchor-") {
			t.Fatalf("job %d output = %q", i, j.Request.OutputPath)
		}
		if seen[j.Request.OutputPath] {
			t.Fatalf("duplicate output %q", j.Request.OutputPath)
		}
		seen[j.Request.OutputPath] = true
		if j.Request.SourcePath != "/src.gguf" || j.Request.ProfileID != "prof1" {
			t.Fatalf("job %d = %+v", i, j.Request)
		}
		if !j.Request.Pure {
			t.Fatalf("anchor job %d did not require --pure", i)
		}
	}
	// Record artifacts, verify hashes, then cleanup.
	var wantHashes []string
	for _, j := range b.Jobs {
		data := []byte("artifact-" + j.Request.OutputPath)
		if err := os.WriteFile(j.Request.OutputPath, data, 0o644); err != nil {
			t.Fatal(err)
		}
		wantHashes = append(wantHashes, mustSHA(t, data))
		a, err := b.RecordArtifact(j.Request.OutputPath)
		if err != nil {
			t.Fatal(err)
		}
		if a.SHA256 != mustSHA(t, data) || a.Bytes != int64(len(data)) {
			t.Fatalf("artifact = %+v", a)
		}
	}
	if err := b.Cleanup(); err != nil {
		t.Fatal(err)
	}
	for _, j := range b.Jobs {
		if _, err := os.Stat(j.Request.OutputPath); !os.IsNotExist(err) {
			t.Fatalf("artifact %s survived cleanup", j.Request.OutputPath)
		}
	}
	// Cleanup is idempotent; missing files tolerated.
	if err := b.Cleanup(); err != nil {
		t.Fatalf("second cleanup: %v", err)
	}
	// Deterministic plan across invocations.
	b2, _ := BuildAnchorBatch("/src.gguf", anchors, dir, "prof1")
	if len(b2.Jobs) != len(b.Jobs) {
		t.Fatalf("nondeterministic job count")
	}
	for i := range b.Jobs {
		if b.Jobs[i].Request.OutputPath != b2.Jobs[i].Request.OutputPath {
			t.Fatalf("nondeterministic output at %d", i)
		}
	}
	// Invalid anchor refused.
	if _, err := BuildAnchorBatch("/s", []core.Anchor{{Kind: "bogus", Name: "x", MinDType: core.DTypeQ8_0}}, dir, "p"); err == nil {
		t.Fatal("invalid anchor accepted")
	}
}

func pplCaps() *Capabilities {
	c := ParseHelp(ToolPerplexity, "/bin/llama-perplexity",
		"usage: llama-perplexity [options]\n  -m FNAME\n  -f FNAME\n  -c N\n  -chunks N\n  -t N\n  -ngl N\n  -s SEED\n  --kl-divergence\n  --kl-divergence-base FNAME\n  --version\nversion: 1.2\n")
	return &c
}

func argvHas(argv []string, tok string) bool {
	for _, a := range argv {
		if a == tok {
			return true
		}
	}
	return false
}

func TestEvalPlansComparable(t *testing.T) {
	cfg := EvalConfig{CorpusPath: "/corpus.txt", CtxSize: 2048, Chunks: 16, Threads: 8, NGPULayers: 0, Seed: 42}
	base, err := PlanBaselineEval(cfg, "/models/base.gguf", "/work/base.logits", pplCaps(), "/bin/lp")
	if err != nil {
		t.Fatal(err)
	}
	cand, err := PlanCandidateEval(cfg, "/models/cand.gguf", "/work/base.logits", pplCaps(), "/bin/lp")
	if err != nil {
		t.Fatal(err)
	}
	if base.Tool != ToolPerplexity || cand.Tool != ToolPerplexity {
		t.Fatal("wrong tool")
	}
	// Both carry the identical comparability settings.
	for _, iv := range []Invocation{base, cand} {
		joined := strings.Join(iv.Argv, " ")
		for _, want := range []string{"-c 2048", "-f /corpus.txt", "-chunks 16", "-t 8", "-ngl 0", "-s 42", "--kl-divergence-base /work/base.logits"} {
			if !strings.Contains(joined, want) {
				t.Fatalf("argv %v missing %q", iv.Argv, want)
			}
		}
	}
	if argvHas(base.Argv, kldFlag) {
		t.Fatalf("baseline argv must not pass %s: %v", kldFlag, base.Argv)
	}
	if !argvHas(cand.Argv, kldFlag) || !argvHas(cand.Argv, kldBaseFlag) {
		t.Fatalf("candidate argv missing KLD flags: %v", cand.Argv)
	}
	if !ComparableReports(base, cand) {
		t.Fatalf("baseline/candidate not comparable:\n%v\n%v", base.Argv, cand.Argv)
	}
	// A differing corpus breaks comparability.
	bad := cfg
	bad.CorpusPath = "/other.txt"
	badBase, err := PlanBaselineEval(bad, "/models/base.gguf", "/work/base.logits", pplCaps(), "/bin/lp")
	if err != nil {
		t.Fatal(err)
	}
	if ComparableReports(badBase, cand) {
		t.Fatal("different corpus reported comparable")
	}
	// Capability gating: old perplexity without --kl-divergence-base.
	oldHelp := ParseHelp(ToolPerplexity, "/bin/old", "usage: llama-perplexity -m FNAME -f FNAME -c N\n")
	if _, err := PlanBaselineEval(cfg, "/m.gguf", "/l", &oldHelp, "/bin/old"); err == nil {
		t.Fatal("missing KLD flag accepted")
	}
	// Missing -chunks advertised likewise refused when needed.
	noChunks := ParseHelp(ToolPerplexity, "/bin/nc", "-m -f -c -t --kl-divergence --kl-divergence-base")
	if _, err := PlanCandidateEval(cfg, "/m.gguf", "/l", &noChunks, "/bin/nc"); err == nil {
		t.Fatal("missing -chunks accepted while requested")
	}
	// Candidate requires --kl-divergence even when --kl-divergence-base is advertised.
	baseOnly := ParseHelp(ToolPerplexity, "/bin/baseonly",
		"usage: llama-perplexity -m FNAME -f FNAME -c N -chunks N -t N -ngl N -s SEED --kl-divergence-base FNAME\n")
	if _, err := PlanCandidateEval(cfg, "/m.gguf", "/l", &baseOnly, "/bin/baseonly"); err == nil {
		t.Fatal("missing --kl-divergence accepted for candidate")
	} else if !strings.Contains(err.Error(), kldFlag) {
		t.Fatalf("candidate missing --kl-divergence: %v", err)
	}
	if _, err := PlanBaselineEval(cfg, "/m.gguf", "/l", &baseOnly, "/bin/baseonly"); err != nil {
		t.Fatalf("baseline rejected without --kl-divergence: %v", err)
	}
	// Invalid config refused.
	if _, err := PlanBaselineEval(EvalConfig{CtxSize: 100}, "/m", "/l", pplCaps(), "/bin/lp"); err == nil {
		t.Fatal("config without corpus accepted")
	}
}

func TestEvalPlansChooseAdvertisedAliases(t *testing.T) {
	cfg := EvalConfig{CorpusPath: "/corpus.txt", CtxSize: 2048, Chunks: 16, Threads: 8, NGPULayers: 0, Seed: 42}
	currentHelp := `usage: llama-perplexity [options]
  --model MODEL
  --file FILE
  --ctx-size N
  --chunks N
  --threads N
  --n-gpu-layers N
  --seed SEED
  --kl-divergence
  --kl-divergence-base FILE
`
	mixedHelp := `usage: llama-perplexity [options]
  --model MODEL
  -f FILE
  --ctx-size N
  -chunks N
  --threads N
  -ngl N
  --seed SEED
  --kl-divergence
  --kl-divergence-base FILE
`
	for _, tc := range []struct {
		name string
		help string
		want []string
	}{
		{"current-long", currentHelp, []string{"--model", "--file", "--ctx-size", "--chunks", "--threads", "--n-gpu-layers", "--seed"}},
		{"legacy-short", "-m MODEL\n-f FILE\n-c N\n-chunks N\n-t N\n-ngl N\n-s SEED\n--kl-divergence\n--kl-divergence-base FILE", []string{"-m", "-f", "-c", "-chunks", "-t", "-ngl", "-s"}},
		{"mixed", mixedHelp, []string{"--model", "-f", "--ctx-size", "-chunks", "--threads", "-ngl", "--seed"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			caps := ParseHelp(ToolPerplexity, "/bin/lp", tc.help)
			iv, err := PlanBaselineEval(cfg, "/model.gguf", "/base.logits", &caps, "/bin/lp")
			if err != nil {
				t.Fatal(err)
			}
			for _, want := range tc.want {
				if !strings.Contains(strings.Join(iv.Argv, " "), want+" ") {
					t.Fatalf("argv %v missing alias %q", iv.Argv, want)
				}
			}
		})
	}

	current := ParseHelp(ToolPerplexity, "/bin/current", currentHelp)
	legacy := pplCaps()
	base, err := PlanBaselineEval(cfg, "/base.gguf", "/base.logits", &current, "/bin/current")
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := PlanCandidateEval(cfg, "/candidate.gguf", "/base.logits", legacy, "/bin/legacy")
	if err != nil {
		t.Fatal(err)
	}
	if !ComparableReports(base, candidate) {
		t.Fatalf("equivalent alias plans not comparable:\n%v\n%v", base.Argv, candidate.Argv)
	}
}

func TestEvalPlansRejectMissingAliases(t *testing.T) {
	cfg := EvalConfig{CorpusPath: "/corpus.txt", CtxSize: 2048, Chunks: 16, Threads: 8, NGPULayers: 0, Seed: 42}
	help := `--model MODEL
-m MODEL
--file FILE
-f FILE
--ctx-size N
-c N
--chunks N
-chunks N
--threads N
-t N
--n-gpu-layers N
-ngl N
--seed SEED
-s SEED
--kl-divergence-base FILE
`
	for _, tc := range []struct {
		name    string
		aliases []string
	}{
		{"model", modelAliases},
		{"file", []string{"--file", "-f"}},
		{"ctx-size", []string{"--ctx-size", "-c"}},
		{"chunks", []string{"--chunks", "-chunks"}},
		{"threads", []string{"--threads", "-t"}},
		{"n-gpu-layers", []string{"--n-gpu-layers", "-ngl"}},
		{"seed", []string{"--seed", "-s"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			missing := help
			for _, alias := range tc.aliases {
				missing = strings.ReplaceAll(missing, alias, "unsupported")
			}
			caps := ParseHelp(ToolPerplexity, "/bin/missing", missing)
			if _, err := PlanBaselineEval(cfg, "/model.gguf", "/base.logits", &caps, "/bin/missing"); err == nil {
				t.Fatalf("missing all %s aliases accepted", tc.name)
			}
		})
	}
}

func TestParseEvalMetrics(t *testing.T) {
	current := `
perplexity: tokenizing
==== Unoptimized F16 ====
perplexity: 5.1234 seconds per pass
Final estimate: PPL = 6.1234 +/- 0.0432
===== KLD metrics =====
Mean KLD: 0.012345
p95 KLD: 0.045678
Max KLD: 0.098765
RMS Δp: 0.002345
Same top: 97.5432%
`
	m, err := ParseEvalMetrics(current)
	if err != nil {
		t.Fatal(err)
	}
	if !m.HasPPL || m.Perplexity != 6.1234 || m.PPLStdErr != 0.0432 {
		t.Fatalf("ppl = %+v", m)
	}
	if !m.HasKLD || !m.HasMeanKLD || m.MeanKLD != 0.012345 || m.P95KLD != 0.045678 || m.MaxKLD != 0.098765 {
		t.Fatalf("kld = %+v", m)
	}
	if !m.HasRMS || m.RMSDeltaP != 0.002345 {
		t.Fatalf("rms = %+v", m)
	}
	if !m.HasSameTop || m.SameTop < 0.975431 || m.SameTop > 0.975433 {
		t.Fatalf("sameTop = %+v", m)
	}

	llamaCPP := "Same top p                     : 84.286 +/- 0.758 %\n"
	m, err = ParseEvalMetrics(llamaCPP)
	if err != nil {
		t.Fatal(err)
	}
	if !m.HasSameTop || m.SameTop < 0.84285 || m.SameTop > 0.84287 {
		t.Fatalf("llama.cpp same top p = %+v", m)
	}

	prefixOverGlobal := "Same top p1: 90.00%\nSame top p: 99.00%\n"
	m, err = ParseEvalMetrics(prefixOverGlobal)
	if err != nil {
		t.Fatal(err)
	}
	if !m.HasSameTop || m.SameTop < 0.8999 || m.SameTop > 0.9001 {
		t.Fatalf("same top p1 should win over global: %+v", m)
	}

	p32OverP1 := "Same top p1: 80.00%\nSame top p32: 88.00%\nSame top p: 99.00%\n"
	m, err = ParseEvalMetrics(p32OverP1)
	if err != nil {
		t.Fatal(err)
	}
	if !m.HasSameTop || m.SameTop < 0.8799 || m.SameTop > 0.8801 {
		t.Fatalf("same top p32 should win over p1/global: %+v", m)
	}

	var tokenLines strings.Builder
	tokenLines.WriteString("Same top p: 50.00%\n")
	for i := 0; i < 40; i++ {
		agree := 1
		if i >= 32 {
			agree = 0
		}
		fmt.Fprintf(&tokenLines, "token %d: kld=0.01 same_top=%d\n", i, agree)
	}
	m, err = ParseEvalMetrics(tokenLines.String())
	if err != nil {
		t.Fatal(err)
	}
	if !m.HasSameTop || m.SameTop < 0.999 || m.SameTop > 1.001 {
		t.Fatalf("prefix-32 token same-top = %+v, want 1.0", m)
	}

	chunks := "Mean KLD: 0.1\nchunk 0: KLD = 0.01\nchunk 1: KLD = 0.02\nchunk 2: KLD = 0.03\nchunk 3: KLD = 0.40\nchunk 4: KLD = 0.50\nchunk 5: KLD = 0.60\nchunk 6: KLD = 0.70\nchunk 7: KLD = 0.80\nchunk 8: KLD = 0.90\nchunk 9: KLD = 1.00\nchunk 10: KLD = 1.10\nchunk 11: KLD = 1.20\nchunk 12: KLD = 1.30\nchunk 13: KLD = 1.40\nchunk 14: KLD = 1.50\nchunk 15: KLD = 1.60\nchunk 16: KLD = 1.70\nchunk 17: KLD = 1.80\nchunk 18: KLD = 1.90\nchunk 19: KLD = 2.00\n"
	m, err = ParseEvalMetrics(chunks)
	if err != nil {
		t.Fatal(err)
	}
	if !m.HasCVaR || m.CVaRKLD < 1.9 {
		t.Fatalf("cvar from chunk samples = %+v", m)
	}
	p95Only := "Mean KLD: 0.1\np95 KLD: 0.5\nMax KLD: 2.0\n"
	m, err = ParseEvalMetrics(p95Only)
	if err != nil {
		t.Fatal(err)
	}
	if m.HasCVaR {
		t.Fatalf("p95-only output synthesized CVaR: %+v", m)
	}

	percentile := "Mean KLD: 0.0125\nMaximum KLD: 0.2\n95.0%   KLD: 9.1853\n"
	m, err = ParseEvalMetrics(percentile)
	if err != nil {
		t.Fatal(err)
	}
	if !m.HasKLD || !m.HasMeanKLD || m.MeanKLD != 0.0125 || m.MaxKLD != 0.2 || m.P95KLD != 9.1853 {
		t.Fatalf("percentile KLD = %+v", m)
	}
	shortPercentile := "95% KLD = 0.25\n"
	m, err = ParseEvalMetrics(shortPercentile)
	if err != nil || !m.HasKLD || m.HasMeanKLD || m.P95KLD != 0.25 || m.MeanKLD != 0 || m.MaxKLD != 0 {
		t.Fatalf("short percentile KLD = %+v, err=%v", m, err)
	}

	legacy := "loading model\nperplexity = 7.5\n"
	m, err = ParseEvalMetrics(legacy)
	if err != nil {
		t.Fatal(err)
	}
	if !m.HasPPL || m.Perplexity != 7.5 || m.HasKLD {
		t.Fatalf("legacy = %+v", m)
	}

	if _, err := ParseEvalMetrics("nothing recognizable\n"); err == nil {
		t.Fatal("garbage parsed")
	}
}

func TestRunProvenance(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "tool")
	corpus := filepath.Join(dir, "corpus.txt")
	im := filepath.Join(dir, "imatrix.bin")
	for p, c := range map[string]string{bin: "BINARY", corpus: "CORPUS", im: "IMATRIX"} {
		if err := os.WriteFile(p, []byte(c), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	rp, err := NewRunProvenance(ToolPerplexity, bin, "b4123", "run-1", corpus, im, time.Unix(1700000000, 0))
	if err != nil {
		t.Fatal(err)
	}
	if rp.BinarySHA256 != mustSHA(t, []byte("BINARY")) {
		t.Fatalf("binary hash = %s", rp.BinarySHA256)
	}
	if rp.CorpusSHA != mustSHA(t, []byte("CORPUS")) {
		t.Fatalf("corpus hash = %s", rp.CorpusSHA)
	}
	if rp.ImatrixSHA != mustSHA(t, []byte("IMATRIX")) {
		t.Fatalf("imatrix hash = %s", rp.ImatrixSHA)
	}
	if rp.Tool != "llama-perplexity" || rp.ToolVersion != "b4123" || rp.RunID != "run-1" {
		t.Fatalf("prov = %+v", rp)
	}
	if err := rp.Provenance.Validate(); err != nil {
		t.Fatal(err)
	}
	// Optional imatrix omitted.
	rp2, err := NewRunProvenance(ToolLlamaQuantize, bin, "b4123", "run-2", corpus, "", time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if rp2.ImatrixSHA != "" {
		t.Fatal("unexpected imatrix digest")
	}
	if rp2.MeasuredAt.IsZero() {
		t.Fatal("timestamp not defaulted")
	}
	if _, err := NewRunProvenance(ToolLlamaQuantize, filepath.Join(dir, "missing"), "", "r", "", "", time.Now()); err == nil {
		t.Fatal("missing binary accepted")
	}
	if _, err := NewRunProvenance(ToolLlamaQuantize, bin, "", "", corpus, "", time.Now()); err == nil {
		t.Fatal("empty run id accepted")
	}
}

func mustSHA(t *testing.T, b []byte) string {
	t.Helper()
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
