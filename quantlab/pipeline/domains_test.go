package pipeline

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"quantlab/core"
	"quantlab/state"
)

func TestDomainEvalPathsFailsWhenRegisteredHoldoutIsMissing(t *testing.T) {
	dir := t.TempDir()
	manifest := `{"version":1,"outputs":{"evaluation-code.txt":"x","evaluation-prose.txt":"y"}}`
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "evaluation-code.txt"), []byte("code"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := domainEvalPaths(dir); err == nil {
		t.Fatal("missing registered domain holdout was silently ignored")
	}
}

func TestExpectedDomainSetDoesNotShrinkWithChangedManifest(t *testing.T) {
	dir := t.TempDir()
	code := filepath.Join(dir, "evaluation-code.txt")
	prose := filepath.Join(dir, "evaluation-prose.txt")
	for _, path := range []string{code, prose} {
		if err := os.WriteFile(path, []byte("holdout"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"),
		[]byte(`{"outputs":{"evaluation-code.txt":"x"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	e := &Engine{Run: &state.Run{Config: state.RunConfig{
		EvalCorpus:        filepath.Join(dir, "evaluation.txt"),
		DomainEvalCorpora: map[string]string{"code": code, "prose": prose},
	}}}
	got, err := e.expectedDomainEvalPaths()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("persisted domain set shrank to %v", got)
	}
}

func TestDomainLogitsUseOneSharedScratchFile(t *testing.T) {
	work := t.TempDir()
	e := &Engine{Run: &state.Run{Config: state.RunConfig{WorkDir: work}}}
	code := e.logitsDomainPath("code")
	prose := e.logitsDomainPath("prose")
	if code != prose {
		t.Fatalf("domain logits paths differ: %q != %q", code, prose)
	}
	if code == e.logitsPath() || code == e.searchLogitsPath() {
		t.Fatalf("shared domain logits path collides with retained baseline: %q", code)
	}
}

func TestEvalProvenanceRejectsLegacyAndDifferentChunkCaches(t *testing.T) {
	base := core.Provenance{
		Tool: "llama-perplexity", ToolVersion: "v1", BinarySHA256: "abc", ProducerSHA256: "producer",
		RunID: "r", SourceSHA: "source", CorpusSHA: "corpus", EvalContext: 512, EvalChunks: 4,
		EvalConfigured: true, MeasuredAt: time.Now(),
	}
	legacy := base
	legacy.BinarySHA256 = ""
	if sameEvalProvenance(legacy, base) {
		t.Fatal("legacy cache without binary identity was accepted")
	}
	screen := base
	screen.EvalChunks = 1
	if sameEvalProvenance(screen, base) {
		t.Fatal("screening measurement was accepted as full validation")
	}
	if !sameEvalProvenance(base, base) {
		t.Fatal("identical evaluation provenance was rejected")
	}
}

func TestEvalableDomainEvalPathsSkipsShortHoldouts(t *testing.T) {
	dir := t.TempDir()
	code := filepath.Join(dir, "evaluation-code.txt")
	long := filepath.Join(dir, "evaluation-long-context.txt")
	if err := os.WriteFile(code, []byte("tiny"), 0o644); err != nil {
		t.Fatal(err)
	}
	payload := make([]byte, minPerplexityCorpusBytes(4096))
	for i := range payload {
		payload[i] = 'a'
	}
	if err := os.WriteFile(long, payload, 0o644); err != nil {
		t.Fatal(err)
	}
	e := &Engine{Run: &state.Run{Config: state.RunConfig{
		CtxSize:    4096,
		EvalCorpus: filepath.Join(dir, "evaluation.txt"),
		DomainEvalCorpora: map[string]string{
			"code":         code,
			"long-context": long,
		},
	}}}
	got, err := e.evalableDomainEvalPaths()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := got["code"]; ok {
		t.Fatal("22 KiB-class holdout was treated as evalable at ctx 4096")
	}
	if got["long-context"] != long {
		t.Fatalf("long-context holdout dropped: %v", got)
	}
}

func TestEvalOutputCorpusTooShort(t *testing.T) {
	msg := "perplexity: you need at least 8192 tokens to evaluate perplexity with a context of 4096"
	if !evalOutputCorpusTooShort(msg) {
		t.Fatal("llama-perplexity short-corpus error was not recognized")
	}
	if evalOutputCorpusTooShort("Mean KLD: 1.2") {
		t.Fatal("normal eval output treated as short corpus")
	}
}

func TestSearchCorpusMustDifferFromFinalEvaluation(t *testing.T) {
	dir := t.TempDir()
	eval := filepath.Join(dir, "evaluation.txt")
	if err := os.WriteFile(eval, []byte("holdout"), 0o644); err != nil {
		t.Fatal(err)
	}
	e := &Engine{Run: &state.Run{Config: state.RunConfig{
		EvalCorpus: eval, SearchCorpus: eval,
	}}}
	if _, ok := e.searchCorpusPath(); ok {
		t.Fatal("final evaluation corpus accepted as tuning holdout")
	}
}
