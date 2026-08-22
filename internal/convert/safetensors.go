package convert

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const maxSTHeader = 64 << 20

// TensorRef locates one tensor inside a safetensors shard without loading it.
type TensorRef struct {
	Name   string
	DType  string // F32, F16, BF16, …
	Shape  []int64
	File   string
	Offset int64 // absolute file offset of payload
	Size   int64
}

type stHeaderEntry struct {
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// IndexDir lists tensors from *.safetensors in dir (and optional index.json).
func IndexDir(dir string) ([]TensorRef, error) {
	var shards []string
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		n := strings.ToLower(e.Name())
		if strings.HasSuffix(n, ".safetensors") && !strings.Contains(n, "optimizer") {
			shards = append(shards, filepath.Join(dir, e.Name()))
		}
	}
	if len(shards) == 0 {
		return nil, fmt.Errorf("no .safetensors files in %s", dir)
	}
	var out []TensorRef
	seen := map[string]bool{}
	for _, p := range shards {
		refs, err := indexShard(p)
		if err != nil {
			return nil, err
		}
		for _, r := range refs {
			if seen[r.Name] {
				continue
			}
			seen[r.Name] = true
			out = append(out, r)
		}
	}
	return out, nil
}

func indexShard(path string) ([]TensorRef, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var hdrLen uint64
	if err := binary.Read(f, binary.LittleEndian, &hdrLen); err != nil {
		return nil, fmt.Errorf("safetensors header length: %w", err)
	}
	if hdrLen == 0 || hdrLen > maxSTHeader {
		return nil, fmt.Errorf("safetensors header length %d is unsafe", hdrLen)
	}
	raw := make([]byte, hdrLen)
	if _, err := io.ReadFull(f, raw); err != nil {
		return nil, fmt.Errorf("safetensors header: %w", err)
	}
	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("safetensors header json: %w", err)
	}
	dataStart := int64(8 + hdrLen)
	var out []TensorRef
	for name, blob := range parsed {
		if name == "__metadata__" {
			continue
		}
		var e stHeaderEntry
		if err := json.Unmarshal(blob, &e); err != nil {
			return nil, fmt.Errorf("tensor %s: %w", name, err)
		}
		begin, end := e.DataOffsets[0], e.DataOffsets[1]
		if end < begin {
			return nil, fmt.Errorf("tensor %s: bad data_offsets", name)
		}
		out = append(out, TensorRef{
			Name:   name,
			DType:  strings.ToUpper(e.DType),
			Shape:  e.Shape,
			File:   path,
			Offset: dataStart + begin,
			Size:   end - begin,
		})
	}
	return out, nil
}

// ReadPayload loads one tensor's raw little-endian bytes.
func ReadPayload(t TensorRef) ([]byte, error) {
	f, err := os.Open(t.File)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	buf := make([]byte, t.Size)
	n, err := f.ReadAt(buf, t.Offset)
	if err != nil && err != io.EOF {
		return nil, err
	}
	if int64(n) != t.Size {
		return nil, fmt.Errorf("tensor %s: short read %d/%d", t.Name, n, t.Size)
	}
	return buf, nil
}

func elemSize(dtype string) int {
	switch strings.ToUpper(dtype) {
	case "F64":
		return 8
	case "F32", "I32", "U32":
		return 4
	case "F16", "BF16", "I16", "U16":
		return 2
	case "F8_E4M3", "F8_E5M2", "I8", "U8", "BOOL":
		return 1
	default:
		return 0
	}
}

type stTensor struct {
	DType string
	Shape []int64
	Data  []byte
}

func writeSafetensors(path string, tensors map[string]stTensor) error {
	header := map[string]any{}
	var payload []byte
	var off int64
	for name, t := range tensors {
		end := off + int64(len(t.Data))
		header[name] = map[string]any{
			"dtype":        t.DType,
			"shape":        t.Shape,
			"data_offsets": []int64{off, end},
		}
		payload = append(payload, t.Data...)
		off = end
	}
	raw, err := json.Marshal(header)
	if err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := binary.Write(f, binary.LittleEndian, uint64(len(raw))); err != nil {
		return err
	}
	if _, err := f.Write(raw); err != nil {
		return err
	}
	_, err = f.Write(payload)
	return err
}
