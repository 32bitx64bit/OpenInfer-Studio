package tensorbank

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"quantlab/core"
)

// ---- synthetic GGUF writer ----

type spec struct {
	name string
	dt   core.DType
	ne   []uint64
}

// synthPayload returns deterministic, distinct payload bytes for a tensor.
func synthPayload(name string, dt core.DType, length int) []byte {
	out := make([]byte, length)
	state := uint32(len(name))
	for i := 0; i < len(name); i++ {
		state = state*31 + uint32(name[i])
	}
	state += uint32(len(string(dt)))
	for i := 0; i < length; i++ {
		state = state*1664525 + 1013904223
		out[i] = byte(state >> 24)
	}
	return out
}

// buildGGUF assembles a complete synthetic GGUF and returns its bytes plus
// each tensor's payload slice (for byte-identity checks).
func buildGGUF(t *testing.T, version, align uint32, kvs []KV, ts []spec) ([]byte, map[string][]byte) {
	t.Helper()
	var meta bytes.Buffer
	var h [24]byte
	binary.LittleEndian.PutUint32(h[0:4], magicGGUF)
	binary.LittleEndian.PutUint32(h[4:8], version)
	binary.LittleEndian.PutUint64(h[8:16], uint64(len(ts)))
	binary.LittleEndian.PutUint64(h[16:24], uint64(len(kvs)))
	meta.Write(h[:])

	meta.Write(EncodeKVs(kvs))

	payloads := make(map[string][]byte, len(ts))
	var offs []uint64
	var body bytes.Buffer
	var cur uint64
	for _, s := range ts {
		var elems uint64 = 1
		for _, d := range s.ne {
			elems *= d
		}
		length, ok := s.dt.ExactBytes(elems)
		if !ok {
			t.Fatalf("no geometry for %s", s.dt)
		}
		aligned := (cur + uint64(align) - 1) / uint64(align) * uint64(align)
		for i := uint64(0); i < aligned-cur; i++ {
			body.WriteByte(0)
		}
		p := synthPayload(s.name, s.dt, int(length))
		payloads[s.name] = p
		offs = append(offs, aligned)
		body.Write(p)
		cur = aligned + length
	}
	for i, s := range ts {
		var l [8]byte
		var f4 [4]byte
		binary.LittleEndian.PutUint64(l[:], uint64(len(s.name)))
		meta.Write(l[:])
		meta.WriteString(s.name)
		binary.LittleEndian.PutUint32(f4[:], uint32(len(s.ne)))
		meta.Write(f4[:])
		for _, d := range s.ne {
			binary.LittleEndian.PutUint64(l[:], d)
			meta.Write(l[:])
		}
		gt, ok := GGMLTypeID(s.dt)
		if !ok {
			t.Fatalf("no ggml id for %s", s.dt)
		}
		binary.LittleEndian.PutUint32(f4[:], gt)
		meta.Write(f4[:])
		binary.LittleEndian.PutUint64(l[:], offs[i])
		meta.Write(l[:])
	}

	dataStart := (uint64(meta.Len()) + uint64(align) - 1) / uint64(align) * uint64(align)
	out := make([]byte, dataStart)
	copy(out, meta.Bytes())
	return append(out, body.Bytes()...), payloads
}

func writeTmp(t *testing.T, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func openParse(t *testing.T, data []byte) (*File, *Source) {
	t.Helper()
	path := writeTmp(t, data)
	s, err := OpenSource(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	f, err := Parse(s)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return f, s
}

func allKVTypes() []KV {
	arr := func(et ValueType, elems ...Value) Value {
		return Value{Type: VTArray, Arr: elems}
	}
	return []KV{
		{Key: "general.architecture", Value: Value{Type: VTString, Scalar: "llama"}},
		{Key: "general.name", Value: Value{Type: VTString, Scalar: "test-model"}},
		{Key: "general.alignment", Value: Value{Type: VTUint32, Scalar: uint32(32)}},
		{Key: "u8", Value: Value{Type: VTUint8, Scalar: uint8(200)}},
		{Key: "i8", Value: Value{Type: VTInt8, Scalar: int8(-100)}},
		{Key: "u16", Value: Value{Type: VTUint16, Scalar: uint16(60000)}},
		{Key: "i16", Value: Value{Type: VTInt16, Scalar: int16(-30000)}},
		{Key: "u32", Value: Value{Type: VTUint32, Scalar: uint32(4000000000)}},
		{Key: "i32", Value: Value{Type: VTInt32, Scalar: int32(-2000000000)}},
		{Key: "f32", Value: Value{Type: VTFloat32, Scalar: float32(3.5)}},
		{Key: "bool.t", Value: Value{Type: VTBool, Scalar: true}},
		{Key: "bool.f", Value: Value{Type: VTBool, Scalar: false}},
		{Key: "u64", Value: Value{Type: VTUint64, Scalar: uint64(18000000000000000000)}},
		{Key: "i64", Value: Value{Type: VTInt64, Scalar: int64(-9000000000000000000)}},
		{Key: "f64", Value: Value{Type: VTFloat64, Scalar: 2.718281828459045}},
		{Key: "arr.str", Value: arr(VTString,
			Value{Type: VTString, Scalar: "a"}, Value{Type: VTString, Scalar: "bb"}, Value{Type: VTString, Scalar: "ccc"})},
		{Key: "arr.u32", Value: arr(VTUint32,
			Value{Type: VTUint32, Scalar: uint32(1)}, Value{Type: VTUint32, Scalar: uint32(2)})},
		{Key: "arr.f32", Value: arr(VTFloat32,
			Value{Type: VTFloat32, Scalar: float32(1.5)}, Value{Type: VTFloat32, Scalar: float32(-2.5)})},
		{Key: "arr.bool", Value: arr(VTBool,
			Value{Type: VTBool, Scalar: true}, Value{Type: VTBool, Scalar: false})},
		{Key: "arr.i64", Value: arr(VTInt64, Value{Type: VTInt64, Scalar: int64(-7)})},
	}
}

func TestParseAllKVScalarAndArrayTypes(t *testing.T) {
	kvs := allKVTypes()
	data, _ := buildGGUF(t, 3, 32, kvs, []spec{
		{"w1", core.DTypeF16, []uint64{64, 8}},
		{"w2", core.DTypeQ8_0, []uint64{64, 8}},
	})
	f, _ := openParse(t, data)
	if len(f.KVs) != len(kvs) {
		t.Fatalf("kv count %d, want %d", len(f.KVs), len(kvs))
	}
	for i, kv := range kvs {
		got, ok := f.Meta(kv.Key)
		if !ok {
			t.Fatalf("missing key %q", kv.Key)
		}
		if got.Type != kv.Value.Type {
			t.Errorf("%s: type %d, want %d", kv.Key, got.Type, kv.Value.Type)
		}
		if len(got.Arr) != len(kv.Value.Arr) {
			t.Fatalf("%s: array len %d, want %d", kv.Key, len(got.Arr), len(kv.Value.Arr))
		}
		for j := range got.Arr {
			if got.Arr[j].Scalar != kv.Value.Arr[j].Scalar {
				t.Errorf("%s[%d]: %#v, want %#v", kv.Key, j, got.Arr[j].Scalar, kv.Value.Arr[j].Scalar)
			}
		}
		if got.Arr == nil && got.Scalar != kv.Value.Scalar {
			t.Errorf("%s: %#v, want %#v", kv.Key, got.Scalar, kv.Value.Scalar)
		}
		_ = i
	}
	if f.ModelID != "test-model" || f.Architecture != "llama" {
		t.Errorf("identity: %q/%q", f.ModelID, f.Architecture)
	}
	if len(f.KVBytes) == 0 || len(f.MetaBytes) < len(f.KVBytes) {
		t.Error("metadata byte capture wrong")
	}
}

func TestParseAlignment32And64(t *testing.T) {
	for _, align := range []uint32{32, 64} {
		kvs := []KV{
			{Key: "general.alignment", Value: Value{Type: VTUint32, Scalar: align}},
		}
		data, _ := buildGGUF(t, 2, align, kvs, []spec{
			{"a", core.DTypeF32, []uint64{100}},
			{"b", core.DTypeF16, []uint64{70, 3}},
		})
		f, _ := openParse(t, data)
		if f.Alignment != align {
			t.Errorf("alignment %d, want %d", f.Alignment, align)
		}
		if f.DataOffset%int64(align) != 0 {
			t.Errorf("data offset %d not aligned to %d", f.DataOffset, align)
		}
		for _, ti := range f.Tensors {
			if ti.RelOffset%uint64(align) != 0 {
				t.Errorf("tensor %s offset %d not aligned to %d", ti.Name, ti.RelOffset, align)
			}
		}
	}
}

// TestParseRejectsUnsafeTensorName proves names from crafted GGUF headers
// that could inject override lines into --tensor-type files are rejected at
// parse time, before any downstream consumer sees them.
func TestParseRejectsUnsafeTensorName(t *testing.T) {
	for _, name := range []string{
		"evil\nfake.tensor Q8_0", // injects an extra override line
		"evil\rweight",
		" leading",
		"trailing ",
		"em bedded",
		"tab\there",
		"ctl\x07bell",
		"", // empty
	} {
		data, _ := buildGGUF(t, 3, 32, anchorKVs("model"), []spec{
			{name, core.DTypeF16, []uint64{16, 4}},
		})
		path := writeTmp(t, data)
		s, err := OpenSource(path)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := Parse(s); err == nil {
			t.Errorf("tensor name %q parsed", name)
		}
		s.Close()
	}
}

func TestParseV2AndV3(t *testing.T) {
	for _, v := range []uint32{2, 3} {
		data, _ := buildGGUF(t, v, 32, nil, []spec{{"x", core.DTypeF32, []uint64{16}}})
		f, _ := openParse(t, data)
		if f.Header.Version != v {
			t.Errorf("version %d, want %d", f.Header.Version, v)
		}
	}
}

func TestParseRejectsMalformed(t *testing.T) {
	good, _ := buildGGUF(t, 3, 32,
		[]KV{{Key: "general.alignment", Value: Value{Type: VTUint32, Scalar: uint32(32)}}},
		[]spec{{"x", core.DTypeF32, []uint64{32}}, {"y", core.DTypeF16, []uint64{32}}})

	cases := []struct {
		name string
		data []byte
	}{
		{"bad magic", append([]byte("JUNKJUNKJUNKJUNKJUNKJUNK"), good[24:]...)},
		{"empty", nil},
		{"short header", good[:16]},
		{"truncated kv", good[:len(good)/2]},
		{"truncated tail", good[:len(good)-8]},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Parse(memReader{tc.data}); err == nil {
				t.Fatal("expected error")
			}
		})
	}

	// v1 rejected
	v1 := append([]byte(nil), good...)
	binary.LittleEndian.PutUint32(v1[4:8], 1)
	if _, err := Parse(memReader{v1}); err == nil {
		t.Error("v1 accepted")
	}
	// v4 rejected
	v4 := append([]byte(nil), good...)
	binary.LittleEndian.PutUint32(v4[4:8], 4)
	if _, err := Parse(memReader{v4}); err == nil {
		t.Error("v4 accepted")
	}
	// unknown GGML type id
	{
		var b bytes.Buffer
		var h [24]byte
		binary.LittleEndian.PutUint32(h[0:4], magicGGUF)
		binary.LittleEndian.PutUint32(h[4:8], 3)
		binary.LittleEndian.PutUint64(h[8:16], 1)
		binary.LittleEndian.PutUint64(h[16:24], 0)
		b.Write(h[:])
		var l [8]byte
		var f4 [4]byte
		binary.LittleEndian.PutUint64(l[:], 1)
		b.Write(l[:])
		b.WriteString("x")
		binary.LittleEndian.PutUint32(f4[:], 1)
		b.Write(f4[:])
		binary.LittleEndian.PutUint64(l[:], 4)
		b.Write(l[:])
		binary.LittleEndian.PutUint32(f4[:], 99) // unknown type
		b.Write(f4[:])
		binary.LittleEndian.PutUint64(l[:], 0)
		b.Write(l[:])
		if _, err := Parse(memReader{b.Bytes()}); err == nil {
			t.Error("unknown ggml type accepted")
		}
	}
	// array of arrays
	{
		var b bytes.Buffer
		var h [24]byte
		binary.LittleEndian.PutUint32(h[0:4], magicGGUF)
		binary.LittleEndian.PutUint32(h[4:8], 3)
		binary.LittleEndian.PutUint64(h[8:16], 1)
		binary.LittleEndian.PutUint64(h[16:24], 1)
		b.Write(h[:])
		var l [8]byte
		var f4 [4]byte
		binary.LittleEndian.PutUint64(l[:], 1)
		b.Write(l[:])
		b.WriteString("k")
		binary.LittleEndian.PutUint32(f4[:], uint32(VTArray))
		b.Write(f4[:])
		binary.LittleEndian.PutUint32(f4[:], uint32(VTArray)) // element type array
		b.Write(f4[:])
		binary.LittleEndian.PutUint64(l[:], 1)
		b.Write(l[:])
		// tensor entry
		binary.LittleEndian.PutUint64(l[:], 1)
		b.Write(l[:])
		b.WriteString("x")
		binary.LittleEndian.PutUint32(f4[:], 1)
		b.Write(f4[:])
		binary.LittleEndian.PutUint64(l[:], 4)
		b.Write(l[:])
		binary.LittleEndian.PutUint32(f4[:], 0)
		b.Write(f4[:])
		binary.LittleEndian.PutUint64(l[:], 0)
		b.Write(l[:])
		if _, err := Parse(memReader{b.Bytes()}); err == nil {
			t.Error("array-of-arrays accepted")
		}
	}
	// unknown metadata value type
	{
		var b bytes.Buffer
		var h [24]byte
		binary.LittleEndian.PutUint32(h[0:4], magicGGUF)
		binary.LittleEndian.PutUint32(h[4:8], 3)
		binary.LittleEndian.PutUint64(h[8:16], 1)
		binary.LittleEndian.PutUint64(h[16:24], 1)
		b.Write(h[:])
		var l [8]byte
		var f4 [4]byte
		binary.LittleEndian.PutUint64(l[:], 1)
		b.Write(l[:])
		b.WriteString("k")
		binary.LittleEndian.PutUint32(f4[:], 100)
		b.Write(f4[:])
		binary.LittleEndian.PutUint32(f4[:], 1)
		b.Write(f4[:])
		binary.LittleEndian.PutUint64(l[:], 1)
		b.Write(l[:])
		b.WriteString("x")
		binary.LittleEndian.PutUint32(f4[:], 1)
		b.Write(f4[:])
		binary.LittleEndian.PutUint64(l[:], 4)
		b.Write(l[:])
		binary.LittleEndian.PutUint32(f4[:], 0)
		b.Write(f4[:])
		binary.LittleEndian.PutUint64(l[:], 0)
		b.Write(l[:])
		if _, err := Parse(memReader{b.Bytes()}); err == nil {
			t.Error("unknown metadata type accepted")
		}
	}
}

func TestParseRejectsSplitGGUF(t *testing.T) {
	for _, key := range []string{"general.split.no", "general.split.count"} {
		kvs := []KV{
			{Key: "general.alignment", Value: Value{Type: VTUint32, Scalar: uint32(32)}},
			{Key: key, Value: Value{Type: VTUint32, Scalar: uint32(1)}},
		}
		data, _ := buildGGUF(t, 3, 32, kvs, []spec{{"x", core.DTypeF32, []uint64{8}}})
		if _, err := Parse(memReader{data}); err == nil {
			t.Errorf("%s accepted", key)
		}
	}
}

func TestParseRejectsOutOfBoundsAndOverlap(t *testing.T) {
	kvs := []KV{{Key: "general.alignment", Value: Value{Type: VTUint32, Scalar: uint32(32)}}}
	data, payloads := buildGGUF(t, 3, 32, kvs, []spec{
		{"x", core.DTypeF32, []uint64{32}},
		{"y", core.DTypeF16, []uint64{32}},
	})
	// truncation so payload exceeds data region
	if _, err := Parse(memReader{data[:len(data)-len(payloads["y"])-4]}); err == nil {
		t.Error("out-of-bounds payload accepted")
	}
	// overlapping offsets: rebuild with y's offset set to x's offset
	{
		var meta bytes.Buffer
		var h [24]byte
		binary.LittleEndian.PutUint32(h[0:4], magicGGUF)
		binary.LittleEndian.PutUint32(h[4:8], 3)
		binary.LittleEndian.PutUint64(h[8:16], 2)
		binary.LittleEndian.PutUint64(h[16:24], 1)
		meta.Write(h[:])
		var l [8]byte
		var f4 [4]byte
		binary.LittleEndian.PutUint64(l[:], uint64(len("general.alignment")))
		meta.Write(l[:])
		meta.WriteString("general.alignment")
		binary.LittleEndian.PutUint32(f4[:], uint32(VTUint32))
		meta.Write(f4[:])
		binary.LittleEndian.PutUint32(f4[:], 32)
		meta.Write(f4[:])
		both := []spec{{"x", core.DTypeF32, []uint64{32}}, {"y", core.DTypeF16, []uint64{32}}}
		for _, s := range both {
			binary.LittleEndian.PutUint64(l[:], uint64(len(s.name)))
			meta.Write(l[:])
			meta.WriteString(s.name)
			binary.LittleEndian.PutUint32(f4[:], uint32(len(s.ne)))
			meta.Write(f4[:])
			for _, d := range s.ne {
				binary.LittleEndian.PutUint64(l[:], d)
				meta.Write(l[:])
			}
			gt, _ := GGMLTypeID(s.dt)
			binary.LittleEndian.PutUint32(f4[:], gt)
			meta.Write(f4[:])
			binary.LittleEndian.PutUint64(l[:], 0) // both at offset 0 -> overlap
			meta.Write(l[:])
		}
		out := make([]byte, meta.Len()+4096)
		copy(out, meta.Bytes())
		if _, err := Parse(memReader{out}); err == nil {
			t.Error("overlapping tensors accepted")
		}
	}
}

func TestGGMLTypeMappingRoundTrip(t *testing.T) {
	want := map[uint32]core.DType{
		0: core.DTypeF32, 1: core.DTypeF16, 2: core.DTypeQ4_0, 3: core.DTypeQ4_1,
		6: core.DTypeQ5_0, 7: core.DTypeQ5_1, 8: core.DTypeQ8_0, 9: core.DTypeQ8_1,
		10: core.DTypeQ2_K, 11: core.DTypeQ3_K, 12: core.DTypeQ4_K_T, 13: core.DTypeQ5_K_T,
		14: core.DTypeQ6_K, 15: core.DTypeQ8_K, 16: core.DTypeIQ2_XXS, 17: core.DTypeIQ2_XS,
		18: core.DTypeIQ3_XXS, 19: core.DTypeIQ1_S, 20: core.DTypeIQ4_NL, 21: core.DTypeIQ3_S,
		22: core.DTypeIQ2_S, 23: core.DTypeIQ4_XS, 24: core.DTypeI8, 25: core.DTypeI16,
		26: core.DTypeI32, 27: core.DTypeI64, 28: core.DTypeF64, 29: core.DTypeIQ1_M,
		30: core.DTypeBF16,
	}
	if len(ggmlToDType) != len(want) || len(dtypeToGGML) != len(want) {
		t.Fatalf("GGML mapping count = forward %d reverse %d, want %d", len(dtypeToGGML), len(ggmlToDType), len(want))
	}
	for id, want := range want {
		got, ok := DTypeFromGGML(id)
		if !ok || got != want {
			t.Errorf("id %d: %v,%v want %v", id, got, ok, want)
		}
		back, ok := GGMLTypeID(want)
		if !ok || back != id {
			t.Errorf("%s: id %d,%v want %d", want, back, ok, id)
		}
	}
	for _, d := range []core.DType{core.DTypeQ4_K_M, core.DTypeQ3_K_S, core.DTypeIQ3_XS} {
		if _, ok := GGMLTypeID(d); ok {
			t.Errorf("%s unexpectedly has a raw GGML type ID", d)
		}
	}
	for _, id := range []uint32{4, 5, 31} {
		if _, ok := DTypeFromGGML(id); ok {
			t.Errorf("unknown id %d accepted", id)
		}
	}
	for d, id := range dtypeToGGML {
		got, ok := DTypeFromGGML(id)
		if !ok || got != d {
			t.Errorf("forward mapping %s -> %d does not reverse: %s,%v", d, id, got, ok)
		}
	}
}

func TestParseBF16TensorType30(t *testing.T) {
	// general.file_type is a GGUF metadata enum, not a ggml_type. Its value
	// must not affect the tensor-table type ID 30.
	data, _ := buildGGUF(t, 3, 32, []KV{
		{Key: "general.file_type", Value: Value{Type: VTUint32, Scalar: uint32(24)}},
	}, []spec{{name: "blk.0.weight", dt: core.DTypeBF16, ne: []uint64{32, 2}}})
	f, _ := openParse(t, data)
	ti, ok := f.FindTensor("blk.0.weight")
	if !ok {
		t.Fatal("BF16 tensor missing")
	}
	if ti.GGMLType != 30 || ti.DType != core.DTypeBF16 || ti.Length != 128 {
		t.Fatalf("BF16 tensor = %+v, want type 30, BF16, 128 bytes", ti)
	}
	fileType, ok := f.Meta("general.file_type")
	if !ok {
		t.Fatal("general.file_type metadata missing")
	}
	if got := fileType.Scalar; got != uint32(24) {
		t.Fatalf("general.file_type = %#v, want metadata value 24", got)
	}
}

func TestHeaderZeroTensorsRejected(t *testing.T) {
	var b bytes.Buffer
	var h [24]byte
	binary.LittleEndian.PutUint32(h[0:4], magicGGUF)
	binary.LittleEndian.PutUint32(h[4:8], 3)
	binary.LittleEndian.PutUint64(h[8:16], 0)
	binary.LittleEndian.PutUint64(h[16:24], 0)
	b.Write(h[:])
	if _, err := Parse(memReader{b.Bytes()}); err == nil {
		t.Error("zero tensor count accepted")
	}
}

func TestAssembleProducesValidBank(t *testing.T) {
	kvs := []KV{
		{Key: "general.architecture", Value: Value{Type: VTString, Scalar: "llama"}},
		{Key: "general.name", Value: Value{Type: VTString, Scalar: "m1"}},
		{Key: "general.alignment", Value: Value{Type: VTUint32, Scalar: uint32(32)}},
	}
	data, _ := buildGGUF(t, 3, 32, kvs, []spec{
		{"w", core.DTypeF16, []uint64{64, 4}},
		{"n", core.DTypeF32, []uint64{64}},
	})
	path := writeTmp(t, data)
	a := NewAssembler()
	s, err := OpenSource(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	bank, err := a.Assemble(s, path, "")
	if err != nil {
		t.Fatal(err)
	}
	if err := bank.Validate(); err != nil {
		t.Fatal(err)
	}
	if len(bank.Tensors) != 2 || bank.ModelID != "m1" {
		t.Fatalf("bank: %+v", bank)
	}
	sha, err := s.SHA256()
	if err != nil || sha != bank.SHA256 {
		t.Errorf("sha %q vs %q (%v)", bank.SHA256, sha, err)
	}
	// second SHA call is cached
	sha2, _ := s.SHA256()
	if sha2 != sha {
		t.Error("sha not cached")
	}
}

// memReader implements Reader over a byte slice.
type memReader struct{ b []byte }

func (m memReader) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 || off > int64(len(m.b)) {
		return 0, os.ErrNotExist
	}
	n := copy(p, m.b[off:])
	if n < len(p) {
		return n, nil
	}
	return n, nil
}
func (m memReader) Size() int64 { return int64(len(m.b)) }
