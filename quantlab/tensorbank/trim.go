package tensorbank

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
)

// Trim writes a valid GGUF to outPath containing only the tensors listed in
// keep, selected from the full anchor at anchorPath. The output preserves the
// anchor's metadata KV bytes verbatim, general.alignment, tensor-table
// ordering (the anchor's own order), dtype, and geometry. Payloads are
// streamed in 1 MiB chunks; memory is bounded. The write is atomic (tmp +
// rename). Context cancellation removes the partial output.
//
// Empty keep is rejected. Every name in keep must exist in the anchor; a
// missing name is a programming error. The result is re-parsed and every
// payload is verified byte-for-byte against the anchor.
func Trim(ctx context.Context, anchorPath string, keep map[string]struct{}, outPath string, progress ProgressFunc) error {
	return NewAssembler().Trim(ctx, anchorPath, keep, outPath, progress)
}

// Trim is the method form of the package-level Trim; see the package-level
// documentation for semantics.
func (a *Assembler) Trim(ctx context.Context, anchorPath string, keep map[string]struct{}, outPath string, progress ProgressFunc) error {
	return a.trim(ctx, anchorPath, keep, outPath, progress, nil, 0)
}

// TrimWithMetadata is Trim, but writes kvs as the GGUF KV section instead of
// copying the anchor's metadata bytes verbatim. Use this to patch scalars
// such as nextn_predict_layers after dropping tensors. kvs must be non-nil.
func TrimWithMetadata(ctx context.Context, anchorPath string, keep map[string]struct{}, outPath string, kvs []KV, progress ProgressFunc) error {
	return NewAssembler().TrimWithMetadata(ctx, anchorPath, keep, outPath, kvs, progress)
}

// TrimWithMetadata is the method form of the package-level TrimWithMetadata.
func (a *Assembler) TrimWithMetadata(ctx context.Context, anchorPath string, keep map[string]struct{}, outPath string, kvs []KV, progress ProgressFunc) error {
	if kvs == nil {
		return fmt.Errorf("tensorbank: nil replacement metadata")
	}
	return a.trim(ctx, anchorPath, keep, outPath, progress, EncodeKVs(kvs), uint64(len(kvs)))
}

func (a *Assembler) trim(ctx context.Context, anchorPath string, keep map[string]struct{}, outPath string, progress ProgressFunc, kvBytes []byte, kvCount uint64) error {
	if len(keep) == 0 {
		return fmt.Errorf("tensorbank: trim keep set is empty")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	abs, absErr := filepath.Abs(outPath)
	if absErr != nil {
		return fmt.Errorf("tensorbank: resolve %q: %w", outPath, absErr)
	}
	anchorAbs, anchorErr := filepath.Abs(anchorPath)
	if anchorErr != nil {
		return fmt.Errorf("tensorbank: resolve %q: %w", anchorPath, anchorErr)
	}
	if anchorAbs == abs {
		return fmt.Errorf("tensorbank: in-place edit rejected: output %q is the anchor", outPath)
	}

	src, err := OpenSource(anchorPath)
	if err != nil {
		return err
	}
	defer src.Close()

	f, err := Parse(src)
	if err != nil {
		return fmt.Errorf("tensorbank: anchor %q: %w", anchorPath, err)
	}

	// Select kept tensors in anchor order; validate that every name in keep
	// exists in the anchor.
	anchorNames := make(map[string]struct{}, len(f.Tensors))
	for _, t := range f.Tensors {
		anchorNames[t.Name] = struct{}{}
	}
	for name := range keep {
		if _, ok := anchorNames[name]; !ok {
			return fmt.Errorf("tensorbank: keep tensor %q not found in anchor", name)
		}
	}

	var picks []TensorInfo
	for _, t := range f.Tensors {
		if _, ok := keep[t.Name]; ok {
			picks = append(picks, t)
		}
	}
	if len(picks) == 0 {
		return fmt.Errorf("tensorbank: no kept tensors found in anchor")
	}

	// Recompute aligned relative offsets in anchor tensor order.
	al := uint64(f.Alignment)
	rel := make([]uint64, len(picks))
	var cur uint64
	var total uint64
	for i, p := range picks {
		cur = alignUp(cur, al)
		rel[i] = cur
		cur += p.Length
		total += p.Length
	}

	tmp := outPath + ".tmp"
	cleanup := func(err error) error {
		os.Remove(tmp)
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	useKV := f.KVBytes
	nkv := uint64(len(f.KVs))
	if kvBytes != nil {
		useKV = kvBytes
		nkv = kvCount
	}

	posHdr := uint64(24) + uint64(len(useKV)) + tensorInfosLenFromInfo(picks)
	padToHdr := alignUp(posHdr, al)
	finalSize := padToHdr + cur

	out, err := a.createDest(tmp, finalSize)
	if err != nil {
		return err
	}
	fail := func(err error) error {
		out.Close()
		return cleanup(err)
	}

	w := &errWriter{w: out}
	var hdr [24]byte
	binary.LittleEndian.PutUint32(hdr[0:4], magicGGUF)
	binary.LittleEndian.PutUint32(hdr[4:8], f.Header.Version)
	binary.LittleEndian.PutUint64(hdr[8:16], uint64(len(picks)))
	binary.LittleEndian.PutUint64(hdr[16:24], nkv)
	w.Write(hdr[:])
	w.Write(useKV)
	for i, p := range picks {
		writeTensorInfo(w, p, rel[i])
	}
	if w.err != nil {
		return fail(fmt.Errorf("tensorbank: write header: %w", w.err))
	}

	// Pad to alignment, then stream payloads.
	pos := uint64(24) + uint64(len(useKV)) + tensorInfosLenFromInfo(picks)
	padTo := alignUp(pos, al)
	if err := writeZeros(out, padTo-pos); err != nil {
		return fail(err)
	}
	pos = padTo

	buf := make([]byte, 1<<20)
	var copied uint64
	srcHash := make([]string, len(picks))
	for i, p := range picks {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		if want := padTo + rel[i]; pos != want {
			if err := writeZeros(out, want-pos); err != nil {
				return fail(err)
			}
			pos = want
		}
		h, err := copyHashedRange(out, src, f.PayloadOffset(p), p.Length, buf)
		if err != nil {
			return fail(fmt.Errorf("tensorbank: copy tensor %q: %w", p.Name, err))
		}
		srcHash[i] = h
		pos += p.Length
		copied += p.Length
		if progress != nil {
			progress(copied, total)
		}
	}
	if err := a.syncFile(out); err != nil {
		return fail(fmt.Errorf("tensorbank: sync: %w", err))
	}
	if err := out.Close(); err != nil {
		return cleanup(fmt.Errorf("tensorbank: close: %w", err))
	}

	if err := a.verifyTrimResult(tmp, picks, rel, srcHash, useKV); err != nil {
		return cleanup(err)
	}

	if err := os.Rename(tmp, outPath); err != nil {
		return cleanup(fmt.Errorf("tensorbank: rename to %q: %w", outPath, err))
	}
	a.syncDir(outPath)
	return nil
}

// verifyTrimResult re-parses the written file and checks every tensor's
// descriptor, the expected metadata section, and every payload hash against
// the parent anchor's source regions. writtenKV is the KV section that was
// emitted (anchor bytes for Trim, a patched encoding for TrimWithMetadata).
func (a *Assembler) verifyTrimResult(path string, picks []TensorInfo, rel []uint64, srcHash []string, writtenKV []byte) error {
	s, err := OpenSource(path)
	if err != nil {
		return err
	}
	defer s.Close()
	f, err := Parse(s)
	if err != nil {
		return fmt.Errorf("tensorbank: trim result re-parse: %w", err)
	}
	if len(f.Tensors) != len(picks) {
		return fmt.Errorf("tensorbank: trim result has %d tensors, want %d", len(f.Tensors), len(picks))
	}
	if !bytes.Equal(f.KVBytes, writtenKV) {
		return fmt.Errorf("tensorbank: trim result metadata differs from expected")
	}
	for i, p := range picks {
		t, ok := f.FindTensor(p.Name)
		if !ok {
			return fmt.Errorf("tensorbank: trim result missing tensor %q", p.Name)
		}
		if t.DType != p.DType || !sameShape(t.Shape, p.Shape) ||
			t.RelOffset != rel[i] || t.Length != p.Length {
			return fmt.Errorf("tensorbank: trim result tensor %q descriptor mismatch", p.Name)
		}
		h, err := hashRange(s, f.PayloadOffset(t), t.Length)
		if err != nil {
			return err
		}
		if h != srcHash[i] {
			return fmt.Errorf("tensorbank: trim result tensor %q payload differs from anchor", p.Name)
		}
	}
	return nil
}

// tensorInfosLenFromInfo sums the serialized tensor-info descriptor sizes for
// a slice of TensorInfo (the standalone counterpart of tensorInfosLen which
// operates on pick).
func tensorInfosLenFromInfo(infos []TensorInfo) uint64 {
	var n uint64
	for _, t := range infos {
		n += 8 + uint64(len(t.Name)) + 4 + 8*uint64(len(t.Shape)) + 4 + 8
	}
	return n
}
