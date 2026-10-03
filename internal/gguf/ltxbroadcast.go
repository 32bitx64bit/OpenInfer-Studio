package gguf

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// LTXBroadcastStatus describes the small LTX-2.5 parameters that GPU
// broadcast operations require as F32. The large quantized matrices stay
// untouched; widening F16/BF16 values to F32 does not change their values.
type LTXBroadcastStatus struct {
	Detected    bool
	TensorCount int
	AddedBytes  int64
	OutputBytes int64
}

func (s LTXBroadcastStatus) Needed() bool { return s.TensorCount > 0 }

func ltxBroadcastName(name string) bool {
	leaf := name[strings.LastIndexByte(name, '.')+1:]
	switch leaf {
	case "keyframes_abs_pos_embedding", "scale_shift_table", "audio_scale_shift_table",
		"prompt_scale_shift_table", "audio_prompt_scale_shift_table",
		"scale_shift_table_a2v_ca_audio", "scale_shift_table_a2v_ca_video":
		return true
	}
	return false
}

func planLTXBroadcast(l *repairLayout) (LTXBroadcastStatus, error) {
	var st LTXBroadcastStatus
	var keyframes, audio, blocks bool
	for _, t := range l.tensors {
		leaf := t.name[strings.LastIndexByte(t.name, '.')+1:]
		keyframes = keyframes || leaf == "keyframes_abs_pos_embedding"
		audio = audio || leaf == "audio_scale_shift_table"
		blocks = blocks || strings.Contains(t.name, "transformer_blocks.")
	}
	if !keyframes || !audio || !blocks {
		return st, nil // A tensor name alone is not an LTX model signature.
	}
	st.Detected = true
	var run int64
	for i := range l.tensors {
		t := &l.tensors[i]
		t.repair = ltxBroadcastName(t.name) && (t.typ == 1 || t.typ == 30)
		if !t.sizeKnown {
			end := uint64(l.fileSize - l.headerSize)
			if i+1 < len(l.tensors) {
				end = l.tensors[i+1].offset
			}
			t.size = end - t.offset // Preserve opaque types, including padding.
		}
		size := t.size
		if t.repair {
			if t.elements == 0 || t.elements > 1<<20 {
				return st, fmt.Errorf("LTX broadcast parameter %q has unexpected size %d", t.name, t.elements)
			}
			st.TensorCount++
			st.AddedBytes += int64(t.size)
			size *= 2
		}
		if size > math.MaxInt64-l.alignment || run > math.MaxInt64-align(int64(size), int64(l.alignment)) {
			return st, fmt.Errorf("LTX compatibility copy size overflows")
		}
		run += align(int64(size), int64(l.alignment))
	}
	if run > math.MaxInt64-l.headerSize {
		return st, fmt.Errorf("LTX compatibility copy size overflows")
	}
	st.OutputBytes = l.headerSize + run
	return st, nil
}

// InspectLTXBroadcast reads only the header. Non-LTX models need no change.
func InspectLTXBroadcast(path string) (LTXBroadcastStatus, error) {
	f, err := os.Open(path)
	if err != nil {
		return LTXBroadcastStatus{}, err
	}
	defer f.Close()
	l, err := readRepairLayout(f)
	if err != nil {
		return LTXBroadcastStatus{}, err
	}
	return planLTXBroadcast(&l)
}

// RepairLTXBroadcast writes a canonical GGUF copy with only the small LTX
// embedding/modulation parameters promoted to F32. All metadata, shapes,
// and other payloads are preserved. The destination must not already exist.
func RepairLTXBroadcast(ctx context.Context, sourcePath, destPath string, progress func(done, total int64)) (st LTXBroadcastStatus, err error) {
	if err := ctx.Err(); err != nil {
		return st, err
	}
	source, err := os.Open(sourcePath)
	if err != nil {
		return st, err
	}
	defer source.Close()
	sourceInfo, err := source.Stat()
	if err != nil {
		return st, err
	}
	l, err := readRepairLayout(source)
	if err != nil {
		return st, err
	}
	st, err = planLTXBroadcast(&l)
	if err != nil {
		return st, err
	}
	if !st.Needed() {
		return st, fmt.Errorf("source has no LTX F16/BF16 broadcast parameters to repair")
	}
	header := make([]byte, l.headerSize)
	if _, err := source.ReadAt(header, 0); err != nil {
		return st, err
	}
	var offset int64
	for _, t := range l.tensors {
		if err := patchHeaderValue(header, t.offsetPos, uint64(offset), 8); err != nil {
			return st, err
		}
		size := int64(t.size)
		if t.repair {
			if err := patchHeaderValue(header, t.typePos, 0, 4); err != nil {
				return st, err
			}
			size *= 2
		}
		offset += align(size, int64(l.alignment))
	}
	if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
		return st, err
	}
	// O_EXCL also prevents truncating the source through a symlink/hard link.
	dest, err := os.OpenFile(destPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return st, err
	}
	keep := false
	defer func() {
		_ = dest.Close()
		if !keep {
			_ = os.Remove(destPath)
		}
	}()
	if _, err := dest.Write(header); err != nil {
		return st, err
	}
	written := l.headerSize
	lastReport := int64(-64 << 20)
	report := func(force bool) {
		if progress != nil && (force || written-lastReport >= 64<<20) {
			lastReport = written
			progress(written, st.OutputBytes)
		}
	}
	report(true)
	for _, t := range l.tensors {
		if err := ctx.Err(); err != nil {
			return st, err
		}
		pos := l.headerSize + int64(t.offset)
		if t.repair {
			if _, err := source.Seek(pos, 0); err != nil {
				return st, err
			}
			var consumed int64
			err = upcastConv(ctx, dest, source, t, &consumed, func(bool) {})
			written += int64(t.size * 2)
		} else {
			err = copyTensor(ctx, dest, source, pos, int64(t.size), &written, report)
		}
		if err != nil {
			return st, fmt.Errorf("tensor %q: %w", t.name, err)
		}
		pad := align(written-l.headerSize, int64(l.alignment)) - (written - l.headerSize)
		if pad > 0 {
			if _, err := dest.Write(make([]byte, pad)); err != nil {
				return st, err
			}
			written += pad
		}
		report(false)
	}
	if written != st.OutputBytes {
		return st, fmt.Errorf("LTX copy is %d bytes, expected %d", written, st.OutputBytes)
	}
	currentInfo, err := os.Stat(sourcePath)
	if err != nil || !os.SameFile(sourceInfo, currentInfo) || currentInfo.Size() != sourceInfo.Size() || !currentInfo.ModTime().Equal(sourceInfo.ModTime()) {
		return st, fmt.Errorf("source changed during LTX compatibility preparation")
	}
	if err := dest.Sync(); err != nil {
		return st, err
	}
	if err := dest.Close(); err != nil {
		return st, err
	}
	issues, _, err := ValidateFile(destPath)
	if err != nil {
		return st, err
	}
	if len(issues) > 0 {
		return st, fmt.Errorf("LTX compatibility copy failed validation: %s", issues[0])
	}
	if err := ctx.Err(); err != nil {
		return st, err
	}
	report(true)
	keep = true
	return st, nil
}
