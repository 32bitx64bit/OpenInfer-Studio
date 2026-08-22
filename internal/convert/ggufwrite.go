package convert

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// GGUF v3 writer. Alignment is 32 bytes (llama.cpp default).
// Call AddKV and PlanTensor first, then WriteHeader, then WriteTensor for each
// planned tensor in order. Tensor payloads are not retained in memory.

const (
	ggufMagic   = 0x46554747
	ggufVersion = 3
	ggufAlign   = 32

	ggufTypeUint8   = 0
	ggufTypeInt8    = 1
	ggufTypeUint16  = 2
	ggufTypeInt16   = 3
	ggufTypeUint32  = 4
	ggufTypeInt32   = 5
	ggufTypeFloat32 = 6
	ggufTypeBool    = 7
	ggufTypeString  = 8
	ggufTypeArray   = 9
	ggufTypeUint64  = 10
	ggufTypeInt64   = 11
	ggufTypeFloat64 = 12

	GGMLF32  = 0
	GGMLF16  = 1
	GGMLBF16 = 30
)

type plannedTensor struct {
	Name   string
	Shape  []int64
	DType  int
	NBytes int64
	Offset uint64
}

type Writer struct {
	f       *os.File
	kv      []KV
	tensors []plannedTensor
	hdrDone bool
	next    int
	dataAt  int64
}

// KV is a GGUF metadata value.
type KV struct {
	Key   string
	Value any
}

func NewWriter(path string) (*Writer, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	return &Writer{f: f}, nil
}

func (w *Writer) AddKV(key string, value any) { w.kv = append(w.kv, KV{Key: key, Value: value}) }

func (w *Writer) PlanTensor(name string, shape []int64, dtype int) error {
	if w.hdrDone {
		return fmt.Errorf("cannot plan tensor after WriteHeader")
	}
	if name == "" {
		return fmt.Errorf("empty tensor name")
	}
	for _, t := range w.tensors {
		if t.Name == name {
			return fmt.Errorf("duplicate tensor %s", name)
		}
	}
	w.tensors = append(w.tensors, plannedTensor{
		Name:   name,
		Shape:  append([]int64(nil), shape...),
		DType:  dtype,
		NBytes: ggmlNBytes(dtype, shape),
	})
	return nil
}

func ggmlNBytes(dtype int, shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	switch dtype {
	case GGMLF32:
		return n * 4
	case GGMLF16, GGMLBF16:
		return n * 2
	default:
		return n
	}
}

func (w *Writer) TensorCount() int { return len(w.tensors) }

func (w *Writer) WriteHeader() error {
	if w.hdrDone {
		return fmt.Errorf("WriteHeader called twice")
	}
	f := w.f
	if err := binary.Write(f, binary.LittleEndian, uint32(ggufMagic)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(ggufVersion)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(len(w.tensors))); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(len(w.kv))); err != nil {
		return err
	}
	for _, kv := range w.kv {
		if err := writeKV(f, kv); err != nil {
			return err
		}
	}
	var off uint64
	for i := range w.tensors {
		off = alignU64(off, ggufAlign)
		w.tensors[i].Offset = off
		off += uint64(w.tensors[i].NBytes)
	}
	for _, t := range w.tensors {
		if err := writeString(f, t.Name); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, uint32(len(t.Shape))); err != nil {
			return err
		}
		for _, d := range t.Shape {
			if err := binary.Write(f, binary.LittleEndian, uint64(d)); err != nil {
				return err
			}
		}
		if err := binary.Write(f, binary.LittleEndian, uint32(t.DType)); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, t.Offset); err != nil {
			return err
		}
	}
	pos, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	pad := alignU64(uint64(pos), ggufAlign) - uint64(pos)
	if pad > 0 {
		if _, err := f.Write(make([]byte, pad)); err != nil {
			return err
		}
	}
	dataAt, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	w.dataAt = dataAt
	w.hdrDone = true
	return nil
}

func (w *Writer) WriteTensor(data []byte) error {
	if !w.hdrDone {
		return fmt.Errorf("WriteHeader first")
	}
	if w.next >= len(w.tensors) {
		return fmt.Errorf("no more planned tensors")
	}
	t := w.tensors[w.next]
	if int64(len(data)) != t.NBytes {
		return fmt.Errorf("tensor %s: payload %d bytes, expected %d", t.Name, len(data), t.NBytes)
	}
	want := w.dataAt + int64(t.Offset)
	pos, err := w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if pos < want {
		if _, err := w.f.Write(make([]byte, want-pos)); err != nil {
			return err
		}
	}
	if _, err := w.f.Write(data); err != nil {
		return err
	}
	w.next++
	return nil
}

func (w *Writer) Close() error {
	if w.f == nil {
		return nil
	}
	var err error
	if w.hdrDone && w.next != len(w.tensors) {
		err = fmt.Errorf("wrote %d/%d tensors", w.next, len(w.tensors))
	}
	cerr := w.f.Close()
	w.f = nil
	if err != nil {
		return err
	}
	return cerr
}

func alignU64(n, a uint64) uint64 {
	if a == 0 {
		return n
	}
	return (n + a - 1) / a * a
}

func writeString(w io.Writer, s string) error {
	b := []byte(s)
	if err := binary.Write(w, binary.LittleEndian, uint64(len(b))); err != nil {
		return err
	}
	_, err := w.Write(b)
	return err
}

func writeKV(w io.Writer, kv KV) error {
	if err := writeString(w, kv.Key); err != nil {
		return err
	}
	return writeValue(w, kv.Value)
}

func writeValue(w io.Writer, v any) error {
	switch x := v.(type) {
	case uint8:
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeUint8)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, x)
	case uint32:
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeUint32)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, x)
	case int32:
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeInt32)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, x)
	case int:
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeInt32)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, int32(x))
	case uint64:
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeUint64)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, x)
	case int64:
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeInt64)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, x)
	case float32:
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeFloat32)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, x)
	case float64:
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeFloat32)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, float32(x))
	case bool:
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeBool)); err != nil {
			return err
		}
		var b uint8
		if x {
			b = 1
		}
		return binary.Write(w, binary.LittleEndian, b)
	case string:
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeString)); err != nil {
			return err
		}
		return writeString(w, x)
	case []string:
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeArray)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeString)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint64(len(x))); err != nil {
			return err
		}
		for _, s := range x {
			if err := writeString(w, s); err != nil {
				return err
			}
		}
		return nil
	case []bool:
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeArray)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeBool)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint64(len(x))); err != nil {
			return err
		}
		for _, b := range x {
			var u uint8
			if b {
				u = 1
			}
			if err := binary.Write(w, binary.LittleEndian, u); err != nil {
				return err
			}
		}
		return nil
	case []int32:
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeArray)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeInt32)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint64(len(x))); err != nil {
			return err
		}
		for _, n := range x {
			if err := binary.Write(w, binary.LittleEndian, n); err != nil {
				return err
			}
		}
		return nil
	case []uint32:
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeArray)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint32(ggufTypeUint32)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint64(len(x))); err != nil {
			return err
		}
		for _, n := range x {
			if err := binary.Write(w, binary.LittleEndian, n); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("unsupported GGUF value type %T", v)
	}
}
