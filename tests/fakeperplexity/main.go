// fakeperplexity stands in for llama-perplexity in tests. A baseline run
// writes a small logits marker; --kl-divergence emits representative current
// llama.cpp mean and tail statistics.
package main

import (
	"fmt"
	"os"
	"strings"
)

func help() string {
	return `usage: llama-perplexity [options]
  -m,   --model FNAME
  -f,   --file FNAME
  -c,   --ctx-size N
  -ngl, --n-gpu-layers N
  -t,   --threads N
        --chunks N
        --kl-divergence
        --kl-divergence-base FNAME
`
}

func main() {
	args := os.Args[1:]
	for _, arg := range args {
		if arg == "--help" || arg == "-h" {
			fmt.Print(help())
			return
		}
	}

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
		case "--kl-divergence":
			compare = true
		case "-c", "--ctx-size", "-ngl", "--n-gpu-layers", "-t", "--threads", "--chunks":
			_ = next()
		default:
			fmt.Fprintf(os.Stderr, "unrecognized argument: %s\n", args[i])
			os.Exit(2)
		}
	}
	if model == "" || corpus == "" || logits == "" {
		fmt.Fprintln(os.Stderr, "model, file, and kl-divergence-base are required")
		os.Exit(2)
	}
	if _, err := os.Stat(corpus); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if !compare {
		if err := os.WriteFile(logits, []byte("_logits_fake\n"), 0o600); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Fprintln(os.Stderr, "Final estimate: PPL = 6.2252 +/- 0.03777")
		return
	}
	if _, err := os.Stat(logits); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if strings.Contains(model, "reject-quality") {
		fmt.Print(`====== Perplexity statistics ======
Mean PPL(Q)                    :  12.8462 +/- 1.6557
Mean PPL(base)                 :  11.7850 +/- 1.1798
Mean    KLD                    :  1.0442 +/- 0.0820
Maximum KLD                    :  30.5924
95.0%   KLD                    :  9.1853
RMS Δp                         :  15.717 +/- 0.961 %
Same top p                     : 84.286 +/- 0.758 %
`)
		return
	}
	fmt.Print(`====== Perplexity statistics ======
Mean PPL(Q)                    :  6.227711 +/- 0.037833
Mean PPL(base)                 :  6.225194 +/- 0.037771
Mean    KLD                    :  0.00002515 +/- 0.00000020
Maximum KLD                    :  0.012206
99.9%   KLD                    :  0.000799
99.0%   KLD                    :  0.000222
95.0%   KLD                    :  0.000121
Median  KLD                    :  0.000013
Minimum KLD                    : -0.000059
`)
}
