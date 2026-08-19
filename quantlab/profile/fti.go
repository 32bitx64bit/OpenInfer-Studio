package profile

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"quantlab/core"
	"quantlab/tensorbank"
)

// DefaultFTIPower raises per-channel importance to this exponent (mean-
// preserving), sharpening the imatrix toward the channels that actually
// drive the loss. 1.0 is a no-op.
const DefaultFTIPower = 1.5

// SharpenValues returns a mean-preserving power-mean of v. The input is
// not mutated.
func SharpenValues(v []float32, power float64) []float32 {
	if len(v) == 0 {
		return v
	}
	if power < 1 {
		power = 1
	}
	if power > 3 {
		power = 3
	}
	out := make([]float32, len(v))
	var sum float64
	n := 0
	for _, x := range v {
		if x > 0 && !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) {
			sum += float64(x)
			n++
		}
	}
	if n == 0 || power == 1 {
		copy(out, v)
		return out
	}
	mean := sum / float64(n)
	if mean <= 0 {
		copy(out, v)
		return out
	}
	for i, x := range v {
		fx := float64(x)
		if fx <= 0 {
			out[i] = x
			continue
		}
		y := mean * math.Pow(fx/mean, power)
		if math.IsNaN(y) || math.IsInf(y, 0) {
			out[i] = x
			continue
		}
		out[i] = float32(y)
	}
	return out
}

// SharpenStats rewrites each tensor's Values in place with SharpenValues.
func SharpenStats(m map[string]ImatrixStats, power float64) {
	for k, st := range m {
		if len(st.Values) == 0 {
			continue
		}
		st.Values = SharpenValues(st.Values, power)
		m[k] = st
	}
}

// WriteSharpenedImatrix copies an imatrix GGUF to dstPath with in_sum2
// tensors sharpened. A legacy .dat source is rewritten as a GGUF imatrix
// so every downstream artifact stays loadable GGUF.
func WriteSharpenedImatrix(srcPath, dstPath string, power float64) error {
	f, err := os.Open(srcPath)
	if err != nil {
		return fmt.Errorf("profile: fti open: %w", err)
	}
	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		f.Close()
		return fmt.Errorf("profile: fti header: %w", err)
	}
	f.Close()
	if binary.LittleEndian.Uint32(magic[:]) != ggufMagic {
		stats, err := LoadImatrix(srcPath)
		if err != nil {
			return err
		}
		if len(stats) == 0 {
			return fmt.Errorf("profile: fti: no imatrix stats in %s", srcPath)
		}
		SharpenStats(stats, power)
		return WriteImatrixGGUF(dstPath, stats)
	}
	src, err := tensorbank.OpenSource(srcPath)
	if err != nil {
		return err
	}
	defer src.Close()
	file, err := tensorbank.Parse(src)
	if err != nil {
		return err
	}
	imatrixTensors := false
	for _, t := range file.Tensors {
		if _, kind := splitImatrixTensor(t.Name); kind == "in_sum2" {
			imatrixTensors = true
			break
		}
	}
	if !imatrixTensors {
		return fmt.Errorf("profile: fti: %s is not an imatrix GGUF", srcPath)
	}
	// Pair in_sum2 with counts so we sharpen the per-sample mean, then
	// write sum2' = x' * count.
	counts := map[string][]float32{}
	for _, t := range file.Tensors {
		name, kind := splitImatrixTensor(t.Name)
		if kind != "counts" || t.DType != core.DTypeF32 {
			continue
		}
		vals, err := readF32(src, file.PayloadOffset(t), t.Length)
		if err != nil {
			return err
		}
		counts[name] = vals
	}
	return tensorbank.Rewrite(src, dstPath, func(w io.Writer, src *tensorbank.Source, abs int64, ti tensorbank.TensorInfo) error {
		name, kind := splitImatrixTensor(ti.Name)
		if kind != "in_sum2" || ti.DType != core.DTypeF32 {
			return tensorbank.CopyPayload(w, src, abs, ti)
		}
		sum2, err := readF32(src, abs, ti.Length)
		if err != nil {
			return err
		}
		cnt := counts[name]
		x := make([]float32, len(sum2))
		row := len(sum2)
		if nmat := len(cnt); nmat > 0 && len(sum2)%nmat == 0 {
			row = len(sum2) / nmat
		}
		for i, v := range sum2 {
			xi := v
			if len(cnt) > 0 && row > 0 {
				ci := i / row
				if ci >= len(cnt) {
					ci = len(cnt) - 1
				}
				if c := cnt[ci]; c > 0 {
					xi = v / c
				}
			}
			x[i] = xi
		}
		xs := SharpenValues(x, power)
		out := make([]byte, ti.Length)
		for i, xv := range xs {
			v := xv
			if len(cnt) > 0 && row > 0 {
				ci := i / row
				if ci >= len(cnt) {
					ci = len(cnt) - 1
				}
				if c := cnt[ci]; c > 0 {
					v = xv * c
				}
			}
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
		}
		_, err = w.Write(out)
		return err
	})
}

// WriteImatrixGGUF writes a llama-imatrix-style GGUF from aggregate stats.
// in_sum2 is Values (or Mean if Values is empty); counts is a single 1.
func WriteImatrixGGUF(path string, stats map[string]ImatrixStats) error {
	type rec struct {
		name string
		vals []float32
		rel  uint64
	}
	names := make([]string, 0, len(stats))
	for n := range stats {
		names = append(names, n)
	}
	sort.Strings(names)
	var recs []rec
	var cur uint64
	add := func(name string, vals []float32) {
		cur = (cur + 31) / 32 * 32
		recs = append(recs, rec{name: name, vals: vals, rel: cur})
		cur += uint64(len(vals)) * 4
	}
	for _, n := range names {
		st := stats[n]
		vals := st.Values
		if len(vals) == 0 {
			if st.Mean <= 0 {
				continue
			}
			vals = []float32{float32(st.Mean)}
		}
		add(n+".in_sum2", vals)
		cnt := make([]float32, 1)
		cnt[0] = 1
		if st.Samples > 0 {
			cnt[0] = float32(st.Samples)
		}
		add(n+".counts", cnt)
	}
	if len(recs) == 0 {
		return fmt.Errorf("profile: fti: no imatrix tensors to write")
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	defer func() {
		f.Close()
		os.Remove(tmp)
	}()
	var hdr [24]byte
	binary.LittleEndian.PutUint32(hdr[0:4], 0x46554747)
	binary.LittleEndian.PutUint32(hdr[4:8], 3)
	binary.LittleEndian.PutUint64(hdr[8:16], uint64(len(recs)))
	binary.LittleEndian.PutUint64(hdr[16:24], 2)
	if _, err := f.Write(hdr[:]); err != nil {
		return err
	}
	writeFTIKVString(f, "general.type", "imatrix")
	writeFTIKVUint32(f, "general.alignment", 32)
	for _, r := range recs {
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], uint64(len(r.name)))
		f.Write(b[:])
		f.Write([]byte(r.name))
		var n [4]byte
		binary.LittleEndian.PutUint32(n[:], 1)
		f.Write(n[:])
		binary.LittleEndian.PutUint64(b[:], uint64(len(r.vals)))
		f.Write(b[:])
		binary.LittleEndian.PutUint32(n[:], 0)
		f.Write(n[:])
		binary.LittleEndian.PutUint64(b[:], r.rel)
		f.Write(b[:])
	}
	metaEnd, _ := f.Seek(0, io.SeekCurrent)
	dataStart := (uint64(metaEnd) + 31) / 32 * 32
	if pad := dataStart - uint64(metaEnd); pad > 0 {
		f.Write(make([]byte, pad))
	}
	for _, r := range recs {
		abs := dataStart + r.rel
		if pos, _ := f.Seek(0, io.SeekCurrent); pos < int64(abs) {
			f.Write(make([]byte, int64(abs)-pos))
		}
		buf := make([]byte, len(r.vals)*4)
		for i, v := range r.vals {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		if _, err := f.Write(buf); err != nil {
			return err
		}
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func writeFTIKVString(w io.Writer, k, v string) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(len(k)))
	w.Write(b[:])
	w.Write([]byte(k))
	var typ [4]byte
	binary.LittleEndian.PutUint32(typ[:], uint32(tensorbank.VTString))
	w.Write(typ[:])
	binary.LittleEndian.PutUint64(b[:], uint64(len(v)))
	w.Write(b[:])
	w.Write([]byte(v))
}

func writeFTIKVUint32(w io.Writer, k string, v uint32) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(len(k)))
	w.Write(b[:])
	w.Write([]byte(k))
	var typ [4]byte
	binary.LittleEndian.PutUint32(typ[:], uint32(tensorbank.VTUint32))
	w.Write(typ[:])
	var val [4]byte
	binary.LittleEndian.PutUint32(val[:], v)
	w.Write(val[:])
}
