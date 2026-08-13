// fakequantize stands in for llama-quantize in tests: --help lists types
// and flags, progress lines match llama.cpp, and the input GGUF is copied
// to the output path.
package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

func help() string {
	return `usage: llama-quantize [--help] [--allow-requantize] [--leave-output-tensor]
       [--pure] [--imatrix] [--include-weights] [--exclude-weights]
       [--output-tensor-type] [--token-embedding-type] [--tensor-type]
       [--tensor-type-file] [--keep-split] [--override-kv] [--dry-run]
       input.gguf output.gguf TYPE [nthreads]

Allowed quantization types:
   0  or  F32     : 32-bit float
   1  or  F16     : 16-bit float
   2  or  Q4_0    :  4.00 bpw
   3  or  Q4_1    :  4.50 bpw
   7  or  Q8_0    :  8.50 bpw
  10  or  Q2_K    :  2.63 bpw
  11  or  Q3_K_S  :  3.50 bpw
  12  or  Q3_K_M  :  3.74 bpw
  13  or  Q3_K_L  :  4.03 bpw
  14  or  Q4_K_S  :  4.50 bpw
  15  or  Q4_K_M  :  4.89 bpw
  16  or  Q5_K_S  :  5.50 bpw
  17  or  Q5_K_M  :  5.70 bpw
  18  or  Q6_K    :  6.56 bpw
  19  or  IQ2_XXS :  2.06 bpw
  20  or  IQ2_XS  :  2.31 bpw
  21  or  Q2_K_S  :  2.50 bpw
  23  or  IQ3_XXS :  3.06 bpw
  25  or  IQ4_NL  :  4.50 bpw
  26  or  IQ3_S   :  3.44 bpw
  27  or  IQ3_M   :  3.66 bpw
  28  or  IQ2_S   :  2.50 bpw
  29  or  IQ2_M   :  2.70 bpw
  30  or  IQ4_XS  :  4.25 bpw
  32  or  BF16    : 16-bit bfloat
  36  or  MXFP4_MOE
                  COPY
`
}

func needsValue(flag string) bool {
	switch flag {
	case "--imatrix", "--output-tensor-type", "--token-embedding-type",
		"--tensor-type", "--tensor-type-file", "--override-kv",
		"--include-weights", "--exclude-weights":
		return true
	}
	return false
}

func main() {
	args := os.Args[1:]
	for _, a := range args {
		if a == "--help" || a == "-h" {
			fmt.Print(help())
			return
		}
	}
	var positionals []string
	dry := false
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--" {
			positionals = append(positionals, args[i+1:]...)
			break
		}
		if a == "--dry-run" {
			dry = true
			continue
		}
		if strings.HasPrefix(a, "-") {
			if needsValue(a) && i+1 < len(args) {
				i++
			}
			continue
		}
		positionals = append(positionals, a)
	}
	if len(positionals) < 3 {
		fmt.Fprintln(os.Stderr, "usage: llama-quantize input output TYPE")
		os.Exit(1)
	}
	in, out, ftype := positionals[0], positionals[1], positionals[2]
	fmt.Fprintf(os.Stderr, "quantizing %s -> %s (%s)\n", in, out, ftype)
	fmt.Fprintf(os.Stderr, "[ 1/3] token_embd.weight - f16 to %s\n", strings.ToLower(ftype))
	fmt.Fprintf(os.Stderr, "[ 2/3] blk.0.attn_q.weight - f16 to %s\n", strings.ToLower(ftype))
	fmt.Fprintf(os.Stderr, "[ 3/3] output.weight - f16 to %s\n", strings.ToLower(ftype))
	if dry {
		fmt.Fprintf(os.Stderr, "dry-run: would write %s\n", out)
		return
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	src, err := os.Open(in)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer src.Close()
	dst, err := os.Create(out)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if _, err := io.Copy(dst, src); err != nil {
		dst.Close()
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := dst.Close(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
