package calibrate

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func gen(domain string, n int, tag string) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("%s record %s %d with some sample prose content here", domain, tag, i)
	}
	return out
}

func baseConfig() *Config {
	return &Config{
		Domains: []DomainSpec{
			{Source: SliceSource(DomainGeneral, gen(DomainGeneral, 40, "a")), Weight: 2},
			{Source: SliceSource(DomainChat, gen(DomainChat, 30, "a"))},
			{Source: SliceSource(DomainCode, gen(DomainCode, 30, "a"))},
		},
		Seed:         42,
		CalibPercent: 60,
		MinRecords:   10,
		MinDomains:   3,
		MinTokens:    100,
	}
}

func TestDeterministicReproduction(t *testing.T) {
	r1, err := collect(context.Background(), baseConfig())
	if err != nil {
		t.Fatal(err)
	}
	r2, err := collect(context.Background(), baseConfig())
	if err != nil {
		t.Fatal(err)
	}
	if len(r1.CalibRecords) != len(r2.CalibRecords) || len(r1.EvalRecords) != len(r2.EvalRecords) {
		t.Fatal("partition sizes differ")
	}
	all1 := append(append([]Record{}, r1.CalibRecords...), r1.EvalRecords...)
	all2 := append(append([]Record{}, r2.CalibRecords...), r2.EvalRecords...)
	for i := range all1 {
		if all1[i].Hash != all2[i].Hash || all1[i].Text != all2[i].Text {
			t.Fatalf("record %d differs between runs", i)
		}
	}
}

func TestDifferentSeed(t *testing.T) {
	c1 := baseConfig()
	c2 := baseConfig()
	c2.Seed = 99
	r1, err := collect(context.Background(), c1)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := collect(context.Background(), c2)
	if err != nil {
		t.Fatal(err)
	}
	// Same record set, but (very likely) different interleave order.
	order := func(recs []Record) string {
		var b strings.Builder
		for _, r := range recs {
			b.WriteString(r.Hash)
		}
		return b.String()
	}
	if order(r1.CalibRecords) == order(r2.CalibRecords) && order(r1.EvalRecords) == order(r2.EvalRecords) {
		t.Fatal("seed had no effect on interleaving")
	}
}

func TestDedupe(t *testing.T) {
	items := append(gen(DomainGeneral, 10, "d"), gen(DomainGeneral, 10, "d")...)
	cfg := &Config{
		Domains:      []DomainSpec{{Source: SliceSource(DomainGeneral, items)}},
		CalibPercent: 50,
	}
	res, err := collect(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if res.Stats.TotalRecords != 10 {
		t.Fatalf("got %d records, want 10 after dedupe", res.Stats.TotalRecords)
	}
}

func TestNoLeakageAcrossPartition(t *testing.T) {
	res, err := collect(context.Background(), baseConfig())
	if err != nil {
		t.Fatal(err)
	}
	calib := map[string]bool{}
	for _, r := range res.CalibRecords {
		calib[r.Hash] = true
	}
	for _, r := range res.EvalRecords {
		if calib[r.Hash] {
			t.Fatalf("record %s leaked into both partitions", r.Hash)
		}
	}
	// Partition is stable: re-checking each record agrees with assignment.
	for _, r := range res.CalibRecords {
		if !partitionFor(r.Hash, "", 60) {
			t.Fatal("calib record fails partition check")
		}
	}
	for _, r := range res.EvalRecords {
		if partitionFor(r.Hash, "", 60) {
			t.Fatal("eval record fails partition check")
		}
	}
}

func TestQuotas(t *testing.T) {
	cfg := baseConfig()
	cfg.Domains[0].Quota = 5
	res, err := collect(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if got := res.Stats.DomainCounts[DomainGeneral]; got != 5 {
		t.Fatalf("general count = %d, want 5", got)
	}
}

func TestGeneralFloorRejection(t *testing.T) {
	m := Mixture{Weights: map[string]float64{DomainChat: 1}}
	if err := m.Validate(); err == nil {
		t.Fatal("expected rejection of mixture without general weight")
	}
	m2 := Mixture{Weights: map[string]float64{DomainGeneral: 0, DomainChat: 1}}
	if err := m2.Validate(); err == nil {
		t.Fatal("expected rejection of zero general weight")
	}
	m3 := Mixture{Weights: map[string]float64{DomainGeneral: 0.1, DomainChat: 1}, MinGeneralWeight: 0.2}
	if err := m3.Validate(); err == nil {
		t.Fatal("expected rejection below floor")
	}
	ok := Mixture{Weights: map[string]float64{DomainGeneral: 0.5, DomainChat: 1}}
	if err := ok.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestTaskCorpora(t *testing.T) {
	tasks := []TaskSpec{
		{Name: "prose", Mixture: Mixture{Weights: map[string]float64{DomainGeneral: 2, DomainChat: 1}}},
		{Name: "coding", Mixture: Mixture{Weights: map[string]float64{DomainGeneral: 1, DomainCode: 3}}},
	}
	factory := func(task string) []DomainSpec {
		return []DomainSpec{
			{Source: SliceSource(DomainGeneral, gen(DomainGeneral, 20, "t"))},
			{Source: SliceSource(DomainChat, gen(DomainChat, 20, "t"))},
			{Source: SliceSource(DomainCode, gen(DomainCode, 20, "t"))},
		}
	}
	out, err := TaskCorpora(context.Background(), tasks, 7, factory, Config{CalibPercent: 50})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := out["prose"].Stats.DomainCounts[DomainCode]; ok {
		t.Fatal("code domain leaked into prose task")
	}
	if out["coding"].Stats.DomainCounts[DomainGeneral] == 0 {
		t.Fatal("general floor missing in coding task output")
	}
}

func TestChatWrapping(t *testing.T) {
	cfg := &Config{
		Domains:      []DomainSpec{{Source: SliceSource(DomainChat, []string{"hello there"})}},
		CalibPercent: 100,
		ChatWrap:     "<|user|>\n{{content}}<|end|>",
	}
	res, err := collect(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	want := "<|user|>\nhello there<|end|>"
	if res.CalibRecords[0].Text != want {
		t.Fatalf("got %q want %q", res.CalibRecords[0].Text, want)
	}

	bad := *cfg
	bad.ChatWrap = "no placeholder"
	if _, err := collect(context.Background(), &bad); err == nil {
		t.Fatal("expected error for template missing placeholder")
	}
}

func TestMalformedAndOversizedRecords(t *testing.T) {
	items := []string{
		"",                         // empty
		"   \n\t  ",                // whitespace only
		"bad\x00control",           // control char stripped -> ok
		string([]byte{0xff, 0xfe}), // invalid UTF-8 -> skipped
		strings.Repeat("x", 100),   // oversized relative to limit -> skipped
		"good record",
	}
	cfg := &Config{
		Domains:        []DomainSpec{{Source: SliceSource(DomainGeneral, items)}},
		CalibPercent:   100,
		MaxRecordBytes: 50,
	}
	res, err := collect(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if res.Stats.TotalRecords != 2 {
		t.Fatalf("got %d records, want 2 (stripped-control + good)", res.Stats.TotalRecords)
	}
	for _, r := range res.CalibRecords {
		if strings.ContainsRune(r.Text, '\x00') {
			t.Fatal("control char survived normalization")
		}
	}
}

func TestCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	blocking := SourceFunc{DomainName: DomainGeneral, NextFn: func(ctx context.Context) (string, error) {
		return "", ctx.Err()
	}}
	cfg := &Config{Domains: []DomainSpec{{Source: blocking}}, CalibPercent: 50}
	if _, err := collect(ctx, cfg); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if _, _, err := Build(ctx, t.TempDir(), baseConfig()); !errors.Is(err, context.Canceled) {
		t.Fatalf("Build: got %v, want context.Canceled", err)
	}
}

func sha256File(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func TestManifestAndOutputHashes(t *testing.T) {
	dir := t.TempDir()
	cfg := baseConfig()
	res, m, err := Build(context.Background(), dir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if m.Version != ManifestVersion {
		t.Fatalf("manifest version %d", m.Version)
	}
	if m.Seed != cfg.Seed {
		t.Fatal("seed mismatch")
	}
	for name, want := range m.Outputs {
		if got := sha256File(t, filepath.Join(dir, name)); got != want {
			t.Fatalf("output %s hash mismatch: got %s want %s", name, got, want)
		}
	}
	// Source hashes/counts consistent with domain mix.
	var srcTotal int
	for _, s := range m.Sources {
		if s.Hash == "" || s.Count == 0 {
			t.Fatalf("bad source entry %+v", s)
		}
		if m.DomainMix[s.Domain] != s.Count {
			t.Fatalf("domain mix mismatch for %s", s.Domain)
		}
		srcTotal += s.Count
	}
	if srcTotal != res.Stats.TotalRecords {
		t.Fatal("source counts do not sum to total")
	}

	// Manifest round-trips as valid versioned JSON.
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var decoded Manifest
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Stats.CalibRecords+decoded.Stats.EvalRecords != decoded.Stats.TotalRecords {
		t.Fatal("stats inconsistent")
	}

	// Deterministic: rebuild in a second dir yields identical manifest.
	dir2 := t.TempDir()
	_, m2, err := Build(context.Background(), dir2, baseConfig())
	if err != nil {
		t.Fatal(err)
	}
	b1, _ := json.Marshal(m)
	b2, _ := json.Marshal(m2)
	if string(b1) != string(b2) {
		t.Fatal("rebuild produced different manifest")
	}

	// No temp files left behind.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Fatalf("temp file left behind: %s", e.Name())
		}
	}
}

func TestValidationFloors(t *testing.T) {
	cfg := &Config{
		Domains:      []DomainSpec{{Source: SliceSource(DomainGeneral, gen(DomainGeneral, 3, "v"))}},
		CalibPercent: 50,
		MinRecords:   10,
	}
	if _, err := collect(context.Background(), cfg); err == nil {
		t.Fatal("expected MinRecords rejection")
	}
}

func TestSourceErrorPropagation(t *testing.T) {
	boom := errors.New("boom")
	src := SourceFunc{DomainName: DomainGeneral, NextFn: func(ctx context.Context) (string, error) {
		return "", boom
	}}
	cfg := &Config{Domains: []DomainSpec{{Source: src}}, CalibPercent: 50}
	if _, err := collect(context.Background(), cfg); !errors.Is(err, boom) {
		t.Fatalf("got %v", err)
	}
}

func TestSliceSourceExhaustion(t *testing.T) {
	s := SliceSource(DomainGeneral, []string{"a"})
	if _, err := s.Next(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Next(context.Background()); err != io.EOF {
		t.Fatalf("got %v, want EOF", err)
	}
}
