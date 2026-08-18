package core

import (
	"fmt"
	"unicode"
)

// MaxTensorNameLen bounds tensor names accepted from GGUF headers. GGUF
// itself allows up to 64 KiB names; real model tensors are far shorter.
const MaxTensorNameLen = 256

// ValidateTensorName enforces the tensor-name safety contract: names come
// from untrusted GGUF headers and are interpolated into "name TYPE" override
// files, so they must be non-empty, bounded, and free of whitespace, control
// characters, and invisible unicode format characters (which would allow
// line injection, argument splitting, or look-alike names).
func ValidateTensorName(name string) error {
	if name == "" {
		return fmt.Errorf("tensor: empty name")
	}
	if len(name) > MaxTensorNameLen {
		return fmt.Errorf("tensor: name length %d exceeds bound %d", len(name), MaxTensorNameLen)
	}
	for _, r := range name {
		if r < 0x20 || r == 0x7f {
			return fmt.Errorf("tensor: name %q contains control character U+%04X", name, r)
		}
		if unicode.IsSpace(r) {
			return fmt.Errorf("tensor: name %q contains whitespace", name)
		}
		if unicode.Is(unicode.Cf, r) {
			return fmt.Errorf("tensor: name %q contains invisible format character U+%04X", name, r)
		}
	}
	return nil
}

// TensorDesc describes one logical tensor inside a GGUF file. Data is never
// held here; Offset/Length address raw bytes within the source file's tensor
// data region.
type TensorDesc struct {
	Name     string   `json:"name"`
	DType    DType    `json:"dtype"`
	Shape    []uint64 `json:"shape"` // row-major dimensions as stored in GGUF
	Offset   uint64   `json:"offset"`
	Length   uint64   `json:"length"`
	Elements uint64   `json:"elements"`
}

func (t TensorDesc) Validate() error {
	if err := ValidateTensorName(t.Name); err != nil {
		return err
	}
	if !t.DType.Valid() {
		return fmt.Errorf("tensor %q: invalid dtype %q", t.Name, t.DType)
	}
	if len(t.Shape) == 0 || len(t.Shape) > 4 {
		return fmt.Errorf("tensor %q: shape rank %d out of range", t.Name, len(t.Shape))
	}
	var elems uint64 = 1
	for i, dim := range t.Shape {
		if dim == 0 {
			return fmt.Errorf("tensor %q: zero dimension at index %d", t.Name, i)
		}
		elems *= dim
	}
	if elems != t.Elements {
		return fmt.Errorf("tensor %q: elements %d does not match shape product %d", t.Name, t.Elements, elems)
	}
	if t.Length == 0 {
		return fmt.Errorf("tensor %q: zero byte length", t.Name)
	}
	return nil
}

// Quantizable reports whether t is a weight tensor eligible for K/IQ
// quantization: a 2D matrix, or a 3D expert stack (ffn_*_exps, which
// llama.cpp quantizes along the shared contiguous dimension). The
// innermost (contiguous) dimension must align to the widest block in both
// cases.
func (t TensorDesc) Quantizable() bool {
	if len(t.Shape) < 2 || len(t.Shape) > 3 {
		return false
	}
	return t.Shape[0]%256 == 0 || t.Shape[0]%32 == 0
}

// TensorBank is the assembled, content-addressable view over the raw tensor
// payloads of one source GGUF. It is the unit every later stage consumes.
type TensorBank struct {
	SourcePath string `json:"sourcePath"`
	ModelID    string `json:"modelID"`
	// SHA256, when set, is the hex digest of the source file, recorded so a
	// resumed run can detect a changed source.
	SHA256 string `json:"sha256,omitempty"`
	// Alignment is the source file's general.alignment, and KVMetadataBytes
	// the raw length of its KV metadata section. Both feed exact final
	// artifact-size computation (see tensorbank.PlannedArtifactSize); zero
	// means "unknown" (older checkpoints) and callers fall back to payload
	// accounting only.
	Alignment     uint64 `json:"alignment,omitempty"`
	KVMetadataLen uint64 `json:"kvMetadataBytes,omitempty"`
	// Arch is the source GGUF general.architecture when recorded
	// (informational; calibration hierarchy keys).
	Arch    string       `json:"arch,omitempty"`
	Tensors []TensorDesc `json:"tensors"`
}

func (b *TensorBank) Validate() error {
	if b.SourcePath == "" {
		return fmt.Errorf("tensorbank: empty source path")
	}
	if len(b.Tensors) == 0 {
		return fmt.Errorf("tensorbank: no tensors")
	}
	seen := make(map[string]struct{}, len(b.Tensors))
	for _, t := range b.Tensors {
		if err := t.Validate(); err != nil {
			return err
		}
		if _, dup := seen[t.Name]; dup {
			return fmt.Errorf("tensorbank: duplicate tensor %q", t.Name)
		}
		seen[t.Name] = struct{}{}
	}
	return nil
}

// Find returns the descriptor for name, or false.
func (b *TensorBank) Find(name string) (TensorDesc, bool) {
	for _, t := range b.Tensors {
		if t.Name == name {
			return t, true
		}
	}
	return TensorDesc{}, false
}

// TotalBytes sums all tensor payload sizes.
func (b *TensorBank) TotalBytes() uint64 {
	var n uint64
	for _, t := range b.Tensors {
		n += t.Length
	}
	return n
}
