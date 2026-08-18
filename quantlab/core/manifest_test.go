package core

import "testing"

func optBank() *TensorBank {
	return &TensorBank{
		SourcePath: "/m.gguf",
		SHA256:     "abc123",
		Tensors: []TensorDesc{
			{Name: "blk.0.ffn_down.weight", DType: DTypeF16, Shape: []uint64{256, 256}, Length: 256 * 256 * 2, Elements: 65536},
			{Name: "blk.0.attn_norm.weight", DType: DTypeF32, Shape: []uint64{256}, Length: 1024, Elements: 256},
		},
	}
}

func TestTensorOptionValidate(t *testing.T) {
	ok := TensorOption{TensorName: "a", Target: DTypeQ4_K_T, Bytes: 144}
	if err := ok.Validate(); err != nil {
		t.Fatalf("valid option rejected: %v", err)
	}
	for _, o := range []TensorOption{
		{TensorName: "", Target: DTypeQ4_K_T, Bytes: 1},
		{TensorName: "a", Target: "NOPE", Bytes: 1},
		{TensorName: "a", Target: DTypeQ4_K_T, Bytes: 0},
		{TensorName: "a", Target: DTypeQ4_K_T, Bytes: 1, PriorLoss: -1},
	} {
		if err := o.Validate(); err == nil {
			t.Errorf("expected error for %+v", o)
		}
	}
}

func TestOptionsForExactBytes(t *testing.T) {
	bank := optBank()
	w, _ := bank.Find("blk.0.ffn_down.weight")
	opts, err := OptionsFor(w, []DType{DTypeQ8_0, DTypeQ4_K_T, DTypeF16})
	if err != nil {
		t.Fatal(err)
	}
	if len(opts) != 2 { // float target skipped
		t.Fatalf("got %d options, want 2", len(opts))
	}
	// 65536 elements: Q8_0 -> 65536/32*34 = 69632; Q4_K -> 65536/256*144 = 36864.
	if opts[0].Bytes != 69632 || opts[1].Bytes != 36864 {
		t.Fatalf("exact bytes = %d/%d, want 69632/36864", opts[0].Bytes, opts[1].Bytes)
	}
}

func TestOptionsForNonQuantizable(t *testing.T) {
	bank := optBank()
	n, _ := bank.Find("blk.0.attn_norm.weight")
	opts, err := OptionsFor(n, []DType{DTypeQ4_K_T})
	if err != nil {
		t.Fatal(err)
	}
	if len(opts) != 1 || opts[0].Target != DTypeF32 || opts[0].Bytes != 1024 {
		t.Fatalf("norm options = %+v, want single F32/1024", opts)
	}
}

func TestManifestForAndValidate(t *testing.T) {
	bank := optBank()
	p := &Profile{
		ID: "p1",
		Assignments: []QuantAssignment{
			{TensorName: "blk.0.ffn_down.weight", Target: DTypeQ4_K_T, BitsPerWeight: 4.5},
			{TensorName: "blk.0.attn_norm.weight", Target: DTypeF32, BitsPerWeight: 32},
		},
	}
	m, err := ManifestFor(p, bank)
	if err != nil {
		t.Fatal(err)
	}
	want := uint64(36864 + 1024)
	if m.TotalBytes != want {
		t.Fatalf("total = %d, want %d", m.TotalBytes, want)
	}
	if m.SourceSHA != "abc123" {
		t.Fatalf("source sha not propagated: %q", m.SourceSHA)
	}
	// Tampered byte cost rejected.
	m.Options[0].Bytes++
	if err := m.Validate(bank); err == nil {
		t.Fatal("geometry mismatch accepted")
	}
	// Missing tensor coverage rejected.
	p2 := &Profile{ID: "p2", Assignments: p.Assignments[:1]}
	if _, err := ManifestFor(p2, bank); err == nil {
		t.Fatal("incomplete coverage accepted")
	}
	// Duplicate option rejected.
	m2 := &SelectionManifest{ProfileID: "x", Options: []TensorOption{
		{TensorName: "a", Target: DTypeQ4_K_T, Bytes: 144},
		{TensorName: "a", Target: DTypeQ8_0, Bytes: 34},
	}, TotalBytes: 178}
	if err := m2.Validate(nil); err == nil {
		t.Fatal("duplicate tensor accepted")
	}
}
