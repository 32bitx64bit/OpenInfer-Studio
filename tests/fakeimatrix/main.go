// fakeimatrix stands in for llama-imatrix in tests: --help lists flags,
// generation writes an output file, combine copies the first --in-file,
// and --show-statistics prints ZD-like scores.
package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
)

func help() string {
	return `usage: llama-imatrix [options]
  -m,    --model FNAME
  -f,    --file FNAME
  -o,    --output-file FNAME
  -ngl,  --n-gpu-layers N
  -t,    --threads N
         --chunks N
         --chunk N
         --parse-special
         --process-output
         --no-ppl
         --in-file FNAME
         --show-statistics
`
}

func main() {
	args := os.Args[1:]
	for _, a := range args {
		if a == "--help" || a == "-h" {
			fmt.Print(help())
			return
		}
	}

	var model, file, out string
	var inFiles []string
	showStats := false
	chunks := 20
	for i := 0; i < len(args); i++ {
		a := args[i]
		val := func() string {
			if i+1 < len(args) {
				i++
				return args[i]
			}
			return ""
		}
		switch a {
		case "-m", "--model":
			model = val()
		case "-f", "--file":
			file = val()
		case "-o", "--output-file":
			out = val()
		case "--in-file":
			inFiles = append(inFiles, val())
		case "--chunks":
			_, _ = fmt.Sscanf(val(), "%d", &chunks)
		case "--show-statistics":
			showStats = true
		case "--n-gpu-layers", "-ngl", "--threads", "-t", "--chunk":
			_ = val()
		}
	}

	if showStats {
		src := ""
		if len(inFiles) > 0 {
			src = inFiles[0]
		} else if model != "" {
			src = model
		}
		fmt.Fprintf(os.Stderr, "statistics for %s\n", src)
		fmt.Println("token_embd.weight  ZD score: 8.2")
		fmt.Println("blk.0.attn_v.weight  zd: 6.1")
		fmt.Println("blk.0.ffn_gate.weight  ZD score 1.4")
		return
	}

	if len(inFiles) >= 2 {
		if out == "" {
			fmt.Fprintln(os.Stderr, "combine requires -o")
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "combining %d imatrices\n", len(inFiles))
		if err := copyFile(inFiles[0], out); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}

	if model == "" || file == "" || out == "" {
		fmt.Fprintln(os.Stderr, "usage: llama-imatrix -m model -f file -o out")
		os.Exit(1)
	}
	if chunks < 1 {
		chunks = 1
	}
	for i := 1; i <= chunks && i <= 5; i++ {
		fmt.Fprintf(os.Stderr, "chunk %d / %d\n", i, chunks)
	}
	src := model
	if st, err := os.Stat(file); err == nil && !st.IsDir() {
		_ = file
	}
	if err := copyFile(src, out); err != nil {
		// Fall back to a tiny marker file so jobs still complete.
		if mkerr := os.MkdirAll(filepath.Dir(out), 0o755); mkerr != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		_ = os.WriteFile(out, []byte("fake-imatrix\n"), 0o644)
	}
	fmt.Fprintf(os.Stderr, "saved imatrix with %d chunks\n", chunks)
}

func copyFile(in, out string) error {
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}
	src, err := os.Open(in)
	if err != nil {
		return err
	}
	defer src.Close()
	dst, err := os.Create(out)
	if err != nil {
		return err
	}
	if _, err := io.Copy(dst, src); err != nil {
		dst.Close()
		return err
	}
	return dst.Close()
}
