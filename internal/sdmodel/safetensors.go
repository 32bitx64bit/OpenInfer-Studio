// Package sdmodel classifies stable-diffusion.cpp model files: what pipeline
// role a weight file plays (VAE, text encoder, LoRA, …), whether it is a
// full single-file checkpoint or a transformer-only component, and whether
// its tensors carry an image or video diffusion signature. It depends only
// on the standard library and internal/gguf, so both internal/models
// (library scan) and internal/mediagen (companion discovery + launch flags)
// can share one source of truth.
package sdmodel

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/gguf"
)

// maxSafetensorsHeader caps how much JSON header we will read into memory.
// Real headers are a few KB to a few MB (many tensors); 100 MiB is far more
// than any legitimate model needs and guards against a corrupt/malicious
// length prefix turning into a huge allocation.
const maxSafetensorsHeader = 100 << 20

// SafetensorsTensorNames reads a .safetensors file's JSON header and returns
// its tensor names, sorted. Only the header (8-byte little-endian length
// prefix + JSON object) is read; tensor payload bytes are never touched.
// The reserved "__metadata__" key is not a tensor and is excluded.
func SafetensorsTensorNames(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var lenBuf [8]byte
	if _, err := io.ReadFull(f, lenBuf[:]); err != nil {
		return nil, fmt.Errorf("reading safetensors header length: %w", err)
	}
	hlen := binary.LittleEndian.Uint64(lenBuf[:])
	if hlen == 0 {
		return nil, fmt.Errorf("empty safetensors header")
	}
	if hlen > maxSafetensorsHeader {
		return nil, fmt.Errorf("safetensors header too large: %d bytes", hlen)
	}
	buf := make([]byte, hlen)
	if _, err := io.ReadFull(f, buf); err != nil {
		return nil, fmt.Errorf("reading safetensors header: %w", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(buf, &raw); err != nil {
		return nil, fmt.Errorf("parsing safetensors header: %w", err)
	}
	names := make([]string, 0, len(raw))
	for k := range raw {
		if k == "__metadata__" {
			continue
		}
		names = append(names, k)
	}
	sort.Strings(names)
	return names, nil
}

// TensorNames returns the tensor names for a diffusion pipeline weight
// file, handling both .safetensors (JSON header) and .gguf (tensor table).
// Other extensions (.ckpt/.pt/.pth pickle or zip containers, .bin pytorch
// pickle) are not header-readable this way and return an error; callers
// should fall back to filename/directory heuristics for those.
func TensorNames(path string) ([]string, error) {
	lower := strings.ToLower(path)
	switch {
	case strings.HasSuffix(lower, ".safetensors"):
		return SafetensorsTensorNames(path)
	case strings.HasSuffix(lower, ".gguf"):
		tensors, _, err := gguf.ListTensors(path)
		if err != nil {
			return nil, err
		}
		names := make([]string, 0, len(tensors))
		for _, t := range tensors {
			names = append(names, t.Name)
		}
		return names, nil
	default:
		return nil, fmt.Errorf("unsupported weight file for tensor listing: %s", filepath.Base(path))
	}
}
