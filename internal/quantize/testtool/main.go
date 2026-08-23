// Command testtool is a real subprocess fixture for quantlab adapter tests.
// It emulates llama-quantize and llama-perplexity based on its argv, producing
// structurally valid anchor GGUFs so quantlab's tensorbank assembler runs.
package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"quantlab/core"
	"quantlab/tensorbank"
)

const help = `usage: quantlab-testtool [options]
  -m, --model MODEL
  -f, --file FILE
  -c, --ctx-size N
  -chunks, --chunks N
  -t, --threads N
  -ngl, --n-gpu-layers N
  -s SEED
  --kl-divergence
  --kl-divergence-base FILE
  --imatrix FILE
  --tensor-type FILE
  --tensor-type-file FILE
  --output-tensor-type TYPE
  --token-embedding-type TYPE
  --pure
  --keep-split
  --dry-run
  --version
types: Q8_0 Q6_K Q5_K Q5_1 Q5_0 Q4_K Q4_1 Q4_0 Q3_K Q2_K IQ4_NL IQ4_XS IQ3_S IQ3_XS IQ3_XXS IQ2_S IQ2_M IQ2_XS IQ2_XXS IQ1_S IQ1_M
version: quantlab-testtool-1.0.0
`

func main() {
	args := os.Args[1:]
	for _, arg := range args {
		if arg == "--help" || arg == "--version" {
			fmt.Print(help)
			return
		}
	}
	for _, arg := range args {
		if arg == "-m" || arg == "--model" {
			if err := perplexity(args); err != nil {
				die(err)
			}
			return
		}
	}
	if err := quantize(args); err != nil {
		die(err)
	}
}

func die(err error) {
	fmt.Fprintln(os.Stderr, "quantlab-testtool:", err)
	os.Exit(1)
}

func perplexity(args []string) error {
	var model, corpus, logits string
	compare := false
	for i := 0; i < len(args); i++ {
		next := func() string {
			if i+1 >= len(args) {
				return ""
			}
			i++
			return args[i]
		}
		switch args[i] {
		case "-m", "--model":
			model = next()
		case "-f", "--file":
			corpus = next()
		case "--kl-divergence-base":
			logits = next()
		case "-c", "--ctx-size", "-chunks", "--chunks", "-t", "--threads", "-ngl", "--n-gpu-layers", "-s":
			_ = next()
		case "--kl-divergence":
			compare = true
		default:
			return fmt.Errorf("unrecognized argument %q", args[i])
		}
	}
	if model == "" || corpus == "" || logits == "" {
		return fmt.Errorf("model, file, and kl-divergence-base are required")
	}
	if _, err := os.Stat(corpus); err != nil {
		return err
	}
	if !compare {
		if err := os.WriteFile(logits, []byte("baseline logits\n"), 0o600); err != nil {
			return err
		}
		fmt.Println("Final estimate: PPL = 6.2252 +/- 0.03777")
		return nil
	}
	if _, err := os.Stat(logits); err != nil {
		return err
	}

	modelBytes, err := os.ReadFile(model)
	if err != nil {
		return err
	}
	if bytes.Contains(modelBytes, []byte("reject-quality")) {
		fmt.Println("Final estimate: PPL = 12.8462 +/- 1.6557")
		fmt.Println("mean KLD: 1.0442")
		fmt.Println("p95 KLD: 9.1853")
		fmt.Println("max KLD: 30.5924")
		return nil
	}
	fmt.Println("Final estimate: PPL = 6.227711 +/- 0.037833")
	fmt.Println("mean KLD: 0.00002515")
	fmt.Println("p95 KLD: 0.000121")
	fmt.Println("max KLD: 0.012206")
	return nil
}

func quantize(args []string) error {
	valueFlags := map[string]bool{
		"--imatrix": true, "--tensor-type": true, "--tensor-type-file": true,
		"--output-tensor-type": true, "--token-embedding-type": true,
	}
	dryRun, pure := false, false
	var positional []string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if strings.HasPrefix(arg, "-") {
			if valueFlags[arg] {
				i++
				continue
			}
			if arg == "--dry-run" {
				dryRun = true
			}
			if arg == "--pure" {
				pure = true
			}
			continue
		}
		positional = append(positional, arg)
	}
	if len(positional) < 3 {
		return fmt.Errorf("want <in> <out> <type> [threads], got %v", positional)
	}
	in, out, typ := positional[0], positional[1], core.DType(positional[2])
	if dryRun {
		st, err := os.Stat(in)
		if err != nil {
			return err
		}
		fmt.Printf("llama_model_quantize_internal: quant size = %d bytes\n", st.Size())
		return nil
	}
	return rewrite(in, out, typ, pure)
}

func quantizable(shape []uint64) bool {
	return len(shape) == 2 && (shape[0]%256 == 0 || shape[0]%32 == 0)
}

func alignUp(n, alignment uint64) uint64 {
	return (n + alignment - 1) / alignment * alignment
}

// rewrite stores every quantizable tensor as target while copying preserved
// tensors byte-for-byte. It matches the anchor contract tensorbank.Build
// requires rather than merely copying the input file.
func rewrite(in, out string, target core.DType, pure bool) error {
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
	if _, ok := tensorbank.GGMLTypeID(target); !ok {
		return fmt.Errorf("unknown dtype %s", target)
	}
	type record struct {
		info tensorbank.TensorInfo
		rel  uint64
		old  tensorbank.TensorInfo
	}
	alignment := uint64(f.Alignment)
	var records []record
	var cursor uint64
	for _, tensor := range f.Tensors {
		next := tensor
		if quantizable(tensor.Shape) {
			next.DType = target
			if !pure && target == core.DTypeQ3_K && strings.Contains(tensor.Name, "ffn_down") {
				next.DType = core.DTypeQ4_K_T
			}
			next.GGMLType, _ = tensorbank.GGMLTypeID(next.DType)
			length, _ := next.DType.ExactBytes(next.Elements)
			next.Length = length
		}
		cursor = alignUp(cursor, alignment)
		records = append(records, record{info: next, rel: cursor, old: tensor})
		cursor += next.Length
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}
	of, err := os.Create(out)
	if err != nil {
		return err
	}
	defer of.Close()
	var header [24]byte
	binary.LittleEndian.PutUint32(header[0:4], 0x46554747)
	binary.LittleEndian.PutUint32(header[4:8], f.Header.Version)
	binary.LittleEndian.PutUint64(header[8:16], uint64(len(records)))
	binary.LittleEndian.PutUint64(header[16:24], uint64(len(f.KVs)))
	if _, err := of.Write(header[:]); err != nil {
		return err
	}
	if _, err := of.Write(f.KVBytes); err != nil {
		return err
	}
	for _, record := range records {
		if err := writeTensorInfo(of, record.info, record.rel); err != nil {
			return err
		}
	}
	metaEnd, err := of.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	dataStart := alignUp(uint64(metaEnd), alignment)
	if err := writeZeros(of, dataStart-uint64(metaEnd)); err != nil {
		return err
	}
	buf := make([]byte, 1<<20)
	for _, record := range records {
		want := dataStart + record.rel
		if pos, err := of.Seek(0, io.SeekCurrent); err != nil {
			return err
		} else if uint64(pos) < want {
			if err := writeZeros(of, want-uint64(pos)); err != nil {
				return err
			}
		}
		if record.info.DType != record.old.DType || record.info.Length != record.old.Length {
			if err := writeZeros(of, record.info.Length); err != nil {
				return err
			}
			continue
		}
		left := record.info.Length
		offset := f.PayloadOffset(record.old)
		for left > 0 {
			n := uint64(len(buf))
			if left < n {
				n = left
			}
			if _, err := s.ReadAt(buf[:n], offset+int64(record.info.Length-left)); err != nil {
				return err
			}
			if _, err := of.Write(buf[:n]); err != nil {
				return err
			}
			left -= n
		}
	}
	return of.Close()
}

func writeTensorInfo(w io.Writer, tensor tensorbank.TensorInfo, rel uint64) error {
	var b8 [8]byte
	var b4 [4]byte
	binary.LittleEndian.PutUint64(b8[:], uint64(len(tensor.Name)))
	if _, err := w.Write(b8[:]); err != nil {
		return err
	}
	if _, err := io.WriteString(w, tensor.Name); err != nil {
		return err
	}
	binary.LittleEndian.PutUint32(b4[:], uint32(len(tensor.Shape)))
	if _, err := w.Write(b4[:]); err != nil {
		return err
	}
	for _, dim := range tensor.Shape {
		binary.LittleEndian.PutUint64(b8[:], dim)
		if _, err := w.Write(b8[:]); err != nil {
			return err
		}
	}
	binary.LittleEndian.PutUint32(b4[:], tensor.GGMLType)
	if _, err := w.Write(b4[:]); err != nil {
		return err
	}
	binary.LittleEndian.PutUint64(b8[:], rel)
	_, err := w.Write(b8[:])
	return err
}

func writeZeros(w io.Writer, n uint64) error {
	var zeros [4096]byte
	for n > 0 {
		chunk := uint64(len(zeros))
		if n < chunk {
			chunk = n
		}
		if _, err := w.Write(zeros[:chunk]); err != nil {
			return err
		}
		n -= chunk
	}
	return nil
}
