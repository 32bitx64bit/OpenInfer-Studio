package gguf

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type comfyFixTensor struct {
	name string
	typ  uint32
	dims []uint64 // as stored
	data []byte
	orig []int32 // PyTorch shape in comfy.gguf.orig_shape.<name>; nil for none
}

func pattern(n int, seed byte) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = seed + byte(i*7)
	}
	return b
}

func fixKVString(b []byte, key, val string) []byte {
	b = binary.LittleEndian.AppendUint64(b, uint64(len(key)))
	b = append(b, key...)
	b = binary.LittleEndian.AppendUint32(b, tString)
	b = binary.LittleEndian.AppendUint64(b, uint64(len(val)))
	return append(b, val...)
}

func fixKVInt32s(b []byte, key string, vals []int32) []byte {
	b = binary.LittleEndian.AppendUint64(b, uint64(len(key)))
	b = append(b, key...)
	b = binary.LittleEndian.AppendUint32(b, tArray)
	b = binary.LittleEndian.AppendUint32(b, tInt32)
	b = binary.LittleEndian.AppendUint64(b, uint64(len(vals)))
	for _, v := range vals {
		b = binary.LittleEndian.AppendUint32(b, uint32(v))
	}
	return b
}

// writeComfyGGUF writes a GGUF the way ComfyUI-GGUF's converter does: stored
// dims, plus an orig_shape key for each reshaped tensor.
func writeComfyGGUF(t *testing.T, tensors []comfyFixTensor) string {
	t.Helper()
	nKV := 1
	var kv []byte
	kv = fixKVString(kv, "general.architecture", "wan")
	for _, ten := range tensors {
		if ten.orig != nil {
			kv = fixKVInt32s(kv, comfyOrigShapePrefix+ten.name, ten.orig)
			nKV++
		}
	}
	var info []byte
	var off uint64
	for _, ten := range tensors {
		info = binary.LittleEndian.AppendUint64(info, uint64(len(ten.name)))
		info = append(info, ten.name...)
		info = binary.LittleEndian.AppendUint32(info, uint32(len(ten.dims)))
		for _, d := range ten.dims {
			info = binary.LittleEndian.AppendUint64(info, d)
		}
		info = binary.LittleEndian.AppendUint32(info, ten.typ)
		info = binary.LittleEndian.AppendUint64(info, off)
		off += uint64(align(int64(len(ten.data)), 32))
	}
	hdr := binary.LittleEndian.AppendUint32(nil, magicGGUF)
	hdr = binary.LittleEndian.AppendUint32(hdr, 3)
	hdr = binary.LittleEndian.AppendUint64(hdr, uint64(len(tensors)))
	hdr = binary.LittleEndian.AppendUint64(hdr, uint64(nKV))
	hdr = append(hdr, kv...)
	hdr = append(hdr, info...)
	out := append(hdr, make([]byte, align(int64(len(hdr)), 32)-int64(len(hdr)))...)
	for _, ten := range tensors {
		out = append(out, ten.data...)
		out = append(out, make([]byte, align(int64(len(ten.data)), 32)-int64(len(ten.data)))...)
	}
	path := filepath.Join(t.TempDir(), "comfy.gguf")
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// payload returns the data bytes of the named tensor in a GGUF.
func payload(t *testing.T, path, name string) []byte {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	l, err := readComfyLayout(f)
	if err != nil {
		t.Fatal(err)
	}
	for _, ten := range l.tensors {
		if ten.name == name {
			buf := make([]byte, ten.size)
			if _, err := f.ReadAt(buf, l.dataStart+int64(ten.offset)); err != nil {
				t.Fatal(err)
			}
			return buf
		}
	}
	t.Fatalf("tensor %q not in %s", name, path)
	return nil
}

func TestRestoreComfyShapes(t *testing.T) {
	q4blocks, err := os.ReadFile(filepath.Join("testdata", "dequant", "q4_k.blocks"))
	if err != nil {
		t.Fatal(err)
	}
	q4ref := readF32s(t, "q4_k.f32")

	flat := pattern(24*4, 1)      // F32 [4,6] stored flat
	plain := pattern(8*3*4, 2)    // F32 [8,3], never reshaped
	kq := q4blocks[:4*144]        // 4 Q4_K blocks stored as [256, 4]
	wide := q4blocks[:3*144]      // 3 blocks = 768 elements, original rows of 384
	conv := pattern(4*4*6*8*2, 5) // F16 stored [4,4,6,8], original 5-D kernel
	src := writeComfyGGUF(t, []comfyFixTensor{
		{name: "flat.weight", typ: 0, dims: []uint64{24}, data: flat, orig: []int32{4, 6}},
		{name: "plain.weight", typ: 0, dims: []uint64{8, 3}, data: plain},
		{name: "kq.weight", typ: 12, dims: []uint64{256, 4}, data: kq, orig: []int32{2, 512}},
		{name: "wide.weight", typ: 12, dims: []uint64{256, 3}, data: wide, orig: []int32{2, 384}},
		{name: "conv.weight", typ: 1, dims: []uint64{4, 4, 6, 8}, data: conv, orig: []int32{8, 3, 2, 4, 4}},
	})

	st, err := InspectComfyShapes(src)
	if err != nil {
		t.Fatal(err)
	}
	if !st.Needed() || st.Reshaped != 4 || st.Widened != 1 || len(st.Unsupported) != 0 {
		t.Fatalf("inspect = %+v", st)
	}

	dest := filepath.Join(t.TempDir(), "out", "restored.gguf")
	var calls int
	got, err := RestoreComfyShapes(context.Background(), src, dest, func(done, total int64) {
		calls++
		if done < 0 || done > total {
			t.Fatalf("progress %d/%d", done, total)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.OutputBytes != st.OutputBytes {
		t.Errorf("OutputBytes %d, inspected %d", got.OutputBytes, st.OutputBytes)
	}
	if fi, _ := os.Stat(dest); fi == nil || fi.Size() != st.OutputBytes {
		t.Errorf("restored file size %v, want %d", fi, st.OutputBytes)
	}
	if calls < 2 {
		t.Errorf("progress called %d times", calls)
	}
	if issues, _, err := ValidateFile(dest); err != nil || len(issues) != 0 {
		t.Fatalf("restored file invalid: %v %v", issues, err)
	}

	tensors, md, err := ListTensors(dest)
	if err != nil {
		t.Fatal(err)
	}
	for k := range md.Raw {
		if strings.HasPrefix(k, comfyOrigShapePrefix) {
			t.Errorf("orig_shape key %s survived", k)
		}
	}
	if md.Raw["general.architecture"] != "wan" {
		t.Errorf("general.architecture = %v", md.Raw["general.architecture"])
	}
	want := map[string]struct {
		typ  uint32
		dims []uint64
	}{
		"flat.weight":  {0, []uint64{6, 4}},
		"plain.weight": {0, []uint64{8, 3}},
		"kq.weight":    {12, []uint64{512, 2}},
		"wide.weight":  {1, []uint64{384, 2}},
		"conv.weight":  {1, []uint64{4, 4, 2, 24}},
	}
	for _, ten := range tensors {
		w, ok := want[ten.Name]
		if !ok {
			t.Fatalf("unexpected tensor %s", ten.Name)
		}
		if ten.TypeID != w.typ || !sameDims(ten.Shape, w.dims) {
			t.Errorf("%s: type %d dims %v, want type %d dims %v", ten.Name, ten.TypeID, ten.Shape, w.typ, w.dims)
		}
		delete(want, ten.Name)
	}
	if len(want) != 0 {
		t.Errorf("tensors missing from the restored file: %v", want)
	}

	// Same bytes, different dims: a metadata-only change.
	for name, data := range map[string][]byte{"flat.weight": flat, "plain.weight": plain, "kq.weight": kq, "conv.weight": conv} {
		if !bytes.Equal(payload(t, dest, name), data) {
			t.Errorf("%s payload changed", name)
		}
	}
	// A widened tensor holds ggml's decoding of the same blocks, as F16.
	wantF16, err := encodeF16(nil, q4ref[:768])
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(payload(t, dest, "wide.weight"), wantF16) {
		t.Error("wide.weight is not the F16 decoding of the source blocks")
	}

	// Nothing left to restore, and the source was not touched.
	if again, err := InspectComfyShapes(dest); err != nil || again.Needed() {
		t.Errorf("restored file still inspects as reshaped: %+v %v", again, err)
	}
	if st2, err := InspectComfyShapes(src); err != nil || st2.Reshaped != 4 {
		t.Errorf("source changed: %+v %v", st2, err)
	}
}

func TestRestoreComfyShapesPlainFile(t *testing.T) {
	src := writeComfyGGUF(t, []comfyFixTensor{{name: "a.weight", typ: 0, dims: []uint64{8}, data: pattern(32, 1)}})
	if st, err := InspectComfyShapes(src); err != nil || st.Needed() {
		t.Fatalf("plain file inspects as reshaped: %+v %v", st, err)
	}
	dest := filepath.Join(t.TempDir(), "x.gguf")
	if _, err := RestoreComfyShapes(context.Background(), src, dest, nil); err == nil {
		t.Fatal("restoring a file with nothing to restore succeeded")
	}
	if _, err := os.Stat(dest); err == nil {
		t.Error("a destination was left behind")
	}
}

func TestRestoreComfyShapesUnsupported(t *testing.T) {
	// Q3_K is not decodable here and its original rows cannot hold its blocks.
	src := writeComfyGGUF(t, []comfyFixTensor{
		{name: "q3.weight", typ: 11, dims: []uint64{256, 3}, data: pattern(3*110, 1), orig: []int32{2, 384}},
	})
	st, err := InspectComfyShapes(src)
	if err != nil || len(st.Unsupported) != 1 {
		t.Fatalf("inspect = %+v %v", st, err)
	}
	dest := filepath.Join(t.TempDir(), "x.gguf")
	_, err = RestoreComfyShapes(context.Background(), src, dest, nil)
	if err == nil || !strings.Contains(err.Error(), "q3.weight") {
		t.Fatalf("err = %v", err)
	}
	if _, err := os.Stat(dest); err == nil {
		t.Error("a destination was left behind")
	}
}

func TestRestoreComfyShapesElementMismatch(t *testing.T) {
	src := writeComfyGGUF(t, []comfyFixTensor{
		{name: "a.weight", typ: 0, dims: []uint64{24}, data: pattern(96, 1), orig: []int32{5, 5}},
	})
	if _, err := InspectComfyShapes(src); err == nil || !strings.Contains(err.Error(), "a.weight") {
		t.Fatalf("err = %v", err)
	}
}

func TestRestoreComfyShapesRefusesInPlaceAndCancel(t *testing.T) {
	src := writeComfyGGUF(t, []comfyFixTensor{
		{name: "a.weight", typ: 0, dims: []uint64{24}, data: pattern(96, 1), orig: []int32{4, 6}},
	})
	if _, err := RestoreComfyShapes(context.Background(), src, src, nil); err == nil {
		t.Error("restored in place")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	dest := filepath.Join(t.TempDir(), "x.gguf")
	if _, err := RestoreComfyShapes(ctx, src, dest, nil); err == nil {
		t.Error("canceled restore succeeded")
	}
	if _, err := os.Stat(dest); err == nil {
		t.Error("a canceled restore left its destination behind")
	}
}

func TestGGMLDimsFromTorch(t *testing.T) {
	for _, c := range []struct{ in, want []uint64 }{
		{[]uint64{96768, 8}, []uint64{8, 96768}},
		{[]uint64{1025, 8}, []uint64{8, 1025}},
		{[]uint64{1152, 3, 2, 16, 16}, []uint64{16, 16, 2, 3456}},
		{[]uint64{7}, []uint64{7}},
	} {
		if got := ggmlDimsFromTorch(c.in); !sameDims(got, c.want) {
			t.Errorf("ggmlDimsFromTorch(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}
