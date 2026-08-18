package tensorbank

import (
	"bytes"
	"encoding/binary"
	"math"
)

// EncodeKVs serializes metadata key/value pairs in GGUF file order, matching
// the KV section captured by Parse as File.KVBytes (no 24-byte header).
func EncodeKVs(kvs []KV) []byte {
	var b bytes.Buffer
	for _, kv := range kvs {
		encodeString(&b, kv.Key)
		encodeValue(&b, kv.Value)
	}
	return b.Bytes()
}

// SetScalar updates every existing key named key to integer value v,
// preserving that key's GGUF scalar type. Keys that are not present are
// left absent; this never inserts metadata. Non-integer values are left
// unchanged. The returned slice is a shallow copy of kvs.
func SetScalar(kvs []KV, key string, v uint64) []KV {
	out := append([]KV(nil), kvs...)
	for i := range out {
		if out[i].Key != key {
			continue
		}
		if nv, ok := uintToScalar(out[i].Value.Type, v); ok {
			out[i].Value.Scalar = nv
		}
	}
	return out
}

func uintToScalar(vt ValueType, v uint64) (any, bool) {
	switch vt {
	case VTUint8:
		return uint8(v), true
	case VTInt8:
		return int8(v), true
	case VTUint16:
		return uint16(v), true
	case VTInt16:
		return int16(v), true
	case VTUint32:
		return uint32(v), true
	case VTInt32:
		return int32(v), true
	case VTUint64:
		return v, true
	case VTInt64:
		return int64(v), true
	case VTBool:
		return v != 0, true
	default:
		return nil, false
	}
}

func encodeString(b *bytes.Buffer, s string) {
	var l [8]byte
	binary.LittleEndian.PutUint64(l[:], uint64(len(s)))
	b.Write(l[:])
	b.WriteString(s)
}

func encodeValue(b *bytes.Buffer, v Value) {
	var t4 [4]byte
	binary.LittleEndian.PutUint32(t4[:], uint32(v.Type))
	b.Write(t4[:])
	if v.Type == VTArray {
		et := VTUint32
		if len(v.Arr) > 0 {
			et = v.Arr[0].Type
		}
		binary.LittleEndian.PutUint32(t4[:], uint32(et))
		b.Write(t4[:])
		var t8 [8]byte
		binary.LittleEndian.PutUint64(t8[:], uint64(len(v.Arr)))
		b.Write(t8[:])
		for _, e := range v.Arr {
			encodeScalarPayload(b, e)
		}
		return
	}
	encodeScalarPayload(b, v)
}

func encodeScalarPayload(b *bytes.Buffer, v Value) {
	var t4 [4]byte
	var t8 [8]byte
	switch v.Type {
	case VTUint8, VTInt8, VTBool:
		b.WriteByte(encodeScalarByte(v.Scalar))
	case VTUint16, VTInt16:
		var b2 [2]byte
		binary.LittleEndian.PutUint16(b2[:], encodeScalarU16(v.Scalar))
		b.Write(b2[:])
	case VTUint32, VTInt32, VTFloat32:
		binary.LittleEndian.PutUint32(t4[:], encodeScalarU32(v.Scalar))
		b.Write(t4[:])
	case VTUint64, VTInt64, VTFloat64:
		binary.LittleEndian.PutUint64(t8[:], encodeScalarU64(v.Scalar))
		b.Write(t8[:])
	case VTString:
		s, _ := v.Scalar.(string)
		encodeString(b, s)
	}
}

func encodeScalarByte(a any) byte {
	switch v := a.(type) {
	case uint8:
		return v
	case int8:
		return byte(v)
	case bool:
		if v {
			return 1
		}
	}
	return 0
}

func encodeScalarU16(a any) uint16 {
	switch v := a.(type) {
	case uint16:
		return v
	case int16:
		return uint16(v)
	}
	return 0
}

func encodeScalarU32(a any) uint32 {
	switch v := a.(type) {
	case uint32:
		return v
	case int32:
		return uint32(v)
	case float32:
		return math.Float32bits(v)
	}
	return 0
}

func encodeScalarU64(a any) uint64 {
	switch v := a.(type) {
	case uint64:
		return v
	case int64:
		return uint64(v)
	case float64:
		return math.Float64bits(v)
	}
	return 0
}
