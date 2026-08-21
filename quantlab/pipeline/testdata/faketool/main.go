// Command faketool is a tiny synthetic llama-quantize / llama-perplexity
// stand-in used by the pipeline end-to-end test. It is compiled by the test
// and executed through the production orchestrate.OSRunner, so the whole
// argv-only execution path is exercised for real.
//
// Behavior:
//
//	--help / --version       print a usage text advertising the full flag set
//	(args contain --model)   perplexity mode: --kl-divergence-base saves logits
//	                      (PPL only); --kl-divergence reads them and prints KLD
//	otherwise             quantize mode: [flags] <in> <out> <type> [threads];
//	                      rewrites the GGUF storing every quantizable tensor
//	                      in the target dtype; --dry-run plans without writing
package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"

	"quantlab/core"
	"quantlab/tensorbank"
)

const help = `usage: faketool [options]
  --model MODEL
  --file FILE
  --ctx-size N
  --chunks N
  --threads N
  --n-gpu-layers N
  --seed SEED
 --kl-divergence
 --kl-divergence-base FILE
 --imatrix FILE
 --tensor-type FILE
 --output-tensor-type TYPE
 --token-embedding-type TYPE
 --pure
 --keep-split
 --dry-run
 --version
version: faketool-1.2.3
types: Q8_0 Q6_K Q5_K Q5_1 Q5_0 Q4_K Q4_1 Q4_0 Q3_K Q2_K
`

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		die("no arguments")
	}
	if args[0] == "--help" || args[0] == "--version" {
		fmt.Print(help)
		return
	}
	hasM := false
	for _, a := range args {
		if a == "--model" {
			hasM = true
		}
	}
	var err error
	if hasM {
		err = perplexity(args)
	} else {
		err = quantize(args)
	}
	if err != nil {
		die(err.Error())
	}
}

func die(msg string) {
	fmt.Fprintln(os.Stderr, "faketool:", msg)
	os.Exit(1)
}

func perplexity(args []string) error {
	logits := ""
	compare := false
	for i := 0; i < len(args); i++ {
		if args[i] == "--kl-divergence-base" && i+1 < len(args) {
			logits = args[i+1]
		}
		if args[i] == "--kl-divergence" {
			compare = true
		}
	}
	if logits == "" {
		return fmt.Errorf("no --kl-divergence-base")
	}
	if compare {
		if _, err := os.Stat(logits); err != nil {
			return err
		}
		fmt.Printf("Final estimate: PPL = 11.7245 +/- 0.1000\n")
		fmt.Printf("Mean KLD: 0.012500\n")
		fmt.Printf("Maximum KLD: 0.030000\n")
		fmt.Printf("95.0%%   KLD: 0.025000\n")
		fmt.Printf("RMS Δp: 0.001000\n")
		fmt.Printf("Same top p: 99.10 %%\n")
		return nil
	}
	if err := os.WriteFile(logits, []byte("fake logits"), 0o644); err != nil {
		return err
	}
	fmt.Printf("Final estimate: PPL = 11.5000 +/- 0.1000\n")
	return nil
}

func quantize(args []string) error {
	valueFlags := map[string]bool{
		"--imatrix": true, "--tensor-type": true,
		"--output-tensor-type": true, "--token-embedding-type": true,
	}
	dry := false
	var pos []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		if strings.HasPrefix(a, "-") {
			if valueFlags[a] {
				i++
				continue
			}
			if a == "--dry-run" {
				dry = true
			}
			continue
		}
		pos = append(pos, a)
	}
	if len(pos) < 3 {
		return fmt.Errorf("want <in> <out> <type> [threads], got %v", pos)
	}
	in, out, typ := pos[0], pos[1], pos[2]
	if dry {
		fmt.Printf("dry-run: %s -> %s as %s\n", in, out, typ)
		return nil
	}
	return rewrite(in, out, core.DType(typ))
}

func quantizable(shape []uint64) bool {
	return len(shape) == 2 && (shape[0]%256 == 0 || shape[0]%32 == 0)
}

func alignUp(n, a uint64) uint64 { return (n + a - 1) / a * a }

// rewrite writes out as a copy of in with every quantizable tensor stored as
// target (zero payload for converted tensors, byte-identical copy otherwise).
func rewrite(in, out string, target core.DType) error {
	s, err := tensorbank.OpenSource(in)
	if err != nil {
		return err
	}
	defer s.Close()
	f, err := tensorbank.Parse(s)
	if err != nil {
		return err
	}
	target = target.BaseTensorType()
	id, ok := tensorbank.GGMLTypeID(target)
	if !ok {
		return fmt.Errorf("unknown dtype %s", target)
	}
	type rec struct {
		ti  tensorbank.TensorInfo
		rel uint64
		old tensorbank.TensorInfo
	}
	al := uint64(f.Alignment)
	var recs []rec
	var cur uint64
	for _, ti := range f.Tensors {
		nt := ti
		if quantizable(ti.Shape) {
			nt.DType = target
			nt.GGMLType = id
			l, _ := target.ExactBytes(nt.Elements)
			nt.Length = l
		}
		cur = alignUp(cur, al)
		recs = append(recs, rec{nt, cur, ti})
		cur += nt.Length
	}
	of, err := os.Create(out)
	if err != nil {
		return err
	}
	var hdr [24]byte
	binary.LittleEndian.PutUint32(hdr[0:4], 0x46554747)
	binary.LittleEndian.PutUint32(hdr[4:8], f.Header.Version)
	binary.LittleEndian.PutUint64(hdr[8:16], uint64(len(recs)))
	binary.LittleEndian.PutUint64(hdr[16:24], uint64(len(f.KVs)))
	of.Write(hdr[:])
	of.Write(f.KVBytes)
	for _, r := range recs {
		var b [8]byte
		var n [4]byte
		binary.LittleEndian.PutUint64(b[:], uint64(len(r.ti.Name)))
		of.Write(b[:])
		io.WriteString(of, r.ti.Name)
		binary.LittleEndian.PutUint32(n[:], uint32(len(r.ti.Shape)))
		of.Write(n[:])
		for _, d := range r.ti.Shape {
			binary.LittleEndian.PutUint64(b[:], d)
			of.Write(b[:])
		}
		binary.LittleEndian.PutUint32(n[:], r.ti.GGMLType)
		of.Write(n[:])
		binary.LittleEndian.PutUint64(b[:], r.rel)
		of.Write(b[:])
	}
	metaLen := 24 + len(f.KVBytes)
	for _, r := range recs {
		metaLen += 8 + len(r.ti.Name) + 4 + 8*len(r.ti.Shape) + 4 + 8
	}
	dataStart := alignUp(uint64(metaLen), al)
	if pos, _ := of.Seek(0, io.SeekCurrent); pos < int64(dataStart) {
		of.Write(make([]byte, int64(dataStart)-pos))
	}
	buf := make([]byte, 1<<20)
	for _, r := range recs {
		abs := dataStart + r.rel
		if pos, _ := of.Seek(0, io.SeekCurrent); pos < int64(abs) {
			of.Write(make([]byte, int64(abs)-pos))
		}
		if r.ti.DType == r.old.DType && r.ti.Length == r.old.Length {
			off := f.PayloadOffset(r.old)
			for left := r.ti.Length; left > 0; {
				n := uint64(len(buf))
				if left < n {
					n = left
				}
				if _, err := s.ReadAt(buf[:n], off+int64(r.ti.Length-left)); err != nil {
					return err
				}
				if _, err := of.Write(buf[:n]); err != nil {
					return err
				}
				left -= n
			}
		} else {
			of.Write(make([]byte, r.ti.Length))
		}
	}
	return of.Close()
}
