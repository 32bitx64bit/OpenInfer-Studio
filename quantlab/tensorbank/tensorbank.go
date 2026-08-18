// Package tensorbank assembles the raw, content-addressable tensor view over
// GGUF sources and writes derived GGUF files by selecting tensor payloads
// from anchor files. Payload bytes are streamed; tensors are never loaded
// wholesale into memory.
package tensorbank

import (
	"encoding/binary"
	"fmt"

	"quantlab/core"
)

// Header is the parsed GGUF fixed header.
type Header struct {
	Magic       uint32
	Version     uint32
	TensorCount uint64
	KVCount     uint64
}

const magicGGUF = 0x46554747 // "GGUF" little-endian

// Reader is the minimal random-access source the assembler needs.
type Reader interface {
	ReadAt(p []byte, off int64) (int, error)
	Size() int64
}

// Assembler parses GGUF sources and builds tensor banks and derived files.
type Assembler struct {
	// Scratch, when true, skips fsync of the output file and its parent
	// directory after rename. Default (false) stays durable for emit/final
	// artifacts. Intermediate search/subset files can set Scratch to cut
	// wall-clock without changing payload bytes.
	Scratch bool
}

// NewAssembler returns a ready-to-use Assembler.
func NewAssembler() *Assembler { return &Assembler{} }

// parseHeader reads and validates the 24-byte GGUF fixed header. GGUF v1 and
// big-endian (GGUF-era "GGJT"/"GGLA") files are rejected with a clear
// unsupported error.
func parseHeader(r Reader) (Header, error) {
	var buf [24]byte
	if r.Size() < int64(len(buf)) {
		return Header{}, fmt.Errorf("tensorbank: source too small (%d bytes)", r.Size())
	}
	if _, err := r.ReadAt(buf[:], 0); err != nil {
		return Header{}, fmt.Errorf("tensorbank: read header: %w", err)
	}
	h := Header{
		Magic:       binary.LittleEndian.Uint32(buf[0:4]),
		Version:     binary.LittleEndian.Uint32(buf[4:8]),
		TensorCount: binary.LittleEndian.Uint64(buf[8:16]),
		KVCount:     binary.LittleEndian.Uint64(buf[16:24]),
	}
	if h.Magic != magicGGUF {
		return Header{}, fmt.Errorf("tensorbank: bad magic 0x%08x (v1 GGJT/GGLA and other formats are not supported)", h.Magic)
	}
	if h.Version == 1 {
		return Header{}, fmt.Errorf("tensorbank: GGUF version 1 is not supported (v2/v3 required)")
	}
	if h.Version < 2 || h.Version > 3 {
		return Header{}, fmt.Errorf("tensorbank: unsupported GGUF version %d", h.Version)
	}
	if h.TensorCount == 0 {
		return Header{}, fmt.Errorf("tensorbank: zero tensors")
	}
	return h, nil
}

// Assemble parses full metadata and returns a validated TensorBank whose
// descriptors address byte ranges in the source's tensor data region. The
// source SHA256 is recorded so resumed runs can detect a changed source.
func (a *Assembler) Assemble(r Reader, sourcePath, modelID string) (*core.TensorBank, error) {
	f, err := Parse(r)
	if err != nil {
		return nil, err
	}
	if modelID == "" {
		modelID = f.ModelID
	}
	sha, err := HashReader(r)
	if err != nil {
		return nil, err
	}
	bank := &core.TensorBank{
		SourcePath:    sourcePath,
		ModelID:       modelID,
		SHA256:        sha,
		Alignment:     uint64(f.Alignment),
		KVMetadataLen: uint64(len(f.KVBytes)),
		Arch:          f.Architecture,
	}
	for _, t := range f.Tensors {
		bank.Tensors = append(bank.Tensors, core.TensorDesc{
			Name:     t.Name,
			DType:    t.DType,
			Shape:    append([]uint64(nil), t.Shape...),
			Offset:   t.RelOffset,
			Length:   t.Length,
			Elements: t.Elements,
		})
	}
	if err := bank.Validate(); err != nil {
		return nil, err
	}
	return bank, nil
}
