package orchestrate

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// EvalConfig fixes every evaluation variable that must be identical between
// a baseline run and any candidate run for their metrics to be comparable:
// corpus, context size, chunk count, threads, GPU offload, and seed.
type EvalConfig struct {
	CorpusPath string `json:"corpusPath"`
	CtxSize    int    `json:"ctxSize"`
	// Chunks, when > 0, bounds evaluation to that many context chunks.
	Chunks int `json:"chunks,omitempty"`
	// Threads, when > 0, fixes the CPU worker count.
	Threads int `json:"threads,omitempty"`
	// NGPULayers, when >= 0, fixes GPU offload; -1 means "do not pass -ngl".
	NGPULayers int `json:"ngpuLayers"`
	// Seed, when non-zero, fixes RNG for any sampling in the harness.
	Seed int64 `json:"seed,omitempty"`
}

func (c EvalConfig) Validate() error {
	if c.CorpusPath == "" {
		return fmt.Errorf("orchestrate: eval config missing corpus path")
	}
	if c.CtxSize <= 0 {
		return fmt.Errorf("orchestrate: eval ctx size must be > 0")
	}
	if c.Chunks < 0 || c.Threads < 0 {
		return fmt.Errorf("orchestrate: negative chunks/threads")
	}
	return nil
}

// evalFlag describes one EvalConfig knob and its llama-perplexity aliases.
// Aliases are ordered from current to legacy spelling.
type evalFlag struct {
	name    string
	aliases []string
	when    func(EvalConfig) bool
	value   func(EvalConfig) string
}

func evalFlagSpecs() []evalFlag {
	return []evalFlag{
		{"ctx-size", []string{"--ctx-size", "-c"}, func(c EvalConfig) bool { return true },
			func(c EvalConfig) string { return strconv.Itoa(c.CtxSize) }},
		{"file", []string{"--file", "-f"}, func(c EvalConfig) bool { return true },
			func(c EvalConfig) string { return c.CorpusPath }},
		{"chunks", []string{"--chunks", "-chunks"}, func(c EvalConfig) bool { return c.Chunks > 0 },
			func(c EvalConfig) string { return strconv.Itoa(c.Chunks) }},
		{"threads", []string{"--threads", "-t"}, func(c EvalConfig) bool { return c.Threads > 0 },
			func(c EvalConfig) string { return strconv.Itoa(c.Threads) }},
		{"n-gpu-layers", []string{"--n-gpu-layers", "-ngl"}, func(c EvalConfig) bool { return c.NGPULayers >= 0 },
			func(c EvalConfig) string { return strconv.Itoa(c.NGPULayers) }},
		{"seed", []string{"--seed", "-s"}, func(c EvalConfig) bool { return c.Seed != 0 },
			func(c EvalConfig) string { return strconv.FormatInt(c.Seed, 10) }},
	}
}

var modelAliases = []string{"--model", "-m"}

func chooseAlias(caps *Capabilities, aliases []string) (string, bool) {
	if caps == nil {
		return aliases[0], true
	}
	for _, alias := range aliases {
		if caps.Has(alias) {
			return alias, true
		}
	}
	return "", false
}

func aliasesText(aliases []string) string { return strings.Join(aliases, " or ") }

// kldBaseFlag is the llama-perplexity switch that names the logits file.
// Without kldFlag, the binary SAVES logits there; with kldFlag, it READS
// them and prints Mean KLD. Passing kldFlag on a baseline run would try to
// read a file that does not exist yet.
const (
	kldBaseFlag = "--kl-divergence-base"
	kldFlag     = "--kl-divergence"
)

// PlanBaselineEval builds the baseline llama-perplexity invocation. It saves
// baseline logits to logitsPath for reuse by every candidate run and must
// never pass --kl-divergence. caps may be nil (current full flag set
// assumed); otherwise every emitted flag must be advertised.
func PlanBaselineEval(cfg EvalConfig, modelPath, logitsPath string, caps *Capabilities, binaryPath string) (Invocation, error) {
	return planEval(cfg, modelPath, logitsPath, caps, binaryPath, false)
}

// PlanCandidateEval builds a candidate invocation that loads the saved
// baseline logits from logitsPath and passes --kl-divergence so the binary
// reports KLD against the baseline. All comparability-relevant settings come
// from the shared EvalConfig and therefore match the baseline exactly.
func PlanCandidateEval(cfg EvalConfig, modelPath, logitsPath string, caps *Capabilities, binaryPath string) (Invocation, error) {
	return planEval(cfg, modelPath, logitsPath, caps, binaryPath, true)
}

func planEval(cfg EvalConfig, modelPath, logitsPath string, caps *Capabilities, binaryPath string, compareKLD bool) (Invocation, error) {
	if err := cfg.Validate(); err != nil {
		return Invocation{}, err
	}
	if modelPath == "" || logitsPath == "" {
		return Invocation{}, fmt.Errorf("orchestrate: eval needs model and logits paths")
	}
	if caps != nil && caps.Tool != ToolPerplexity {
		return Invocation{}, fmt.Errorf("orchestrate: capabilities are for %s, not %s", caps.Tool, ToolPerplexity)
	}
	modelFlag, ok := chooseAlias(caps, modelAliases)
	if !ok {
		return Invocation{}, fmt.Errorf("orchestrate: %s does not advertise %s", binaryPath, aliasesText(modelAliases))
	}
	argv := []string{modelFlag, modelPath}
	for _, spec := range evalFlagSpecs() {
		if !spec.when(cfg) {
			continue
		}
		flag, ok := chooseAlias(caps, spec.aliases)
		if !ok {
			return Invocation{}, fmt.Errorf("orchestrate: %s does not advertise %s; cannot honor eval config", binaryPath, aliasesText(spec.aliases))
		}
		argv = append(argv, flag, spec.value(cfg))
	}
	if caps != nil && !caps.Has(kldBaseFlag) {
		return Invocation{}, fmt.Errorf("orchestrate: %s does not advertise %s", binaryPath, kldBaseFlag)
	}
	argv = append(argv, kldBaseFlag, logitsPath)
	if compareKLD {
		if caps != nil && !caps.Has(kldFlag) {
			return Invocation{}, fmt.Errorf("orchestrate: %s does not advertise %s", binaryPath, kldFlag)
		}
		argv = append(argv, kldFlag)
	}
	iv := Invocation{Tool: ToolPerplexity, Path: binaryPath, Argv: argv}
	if err := iv.Validate(); err != nil {
		return Invocation{}, err
	}
	return iv, nil
}

// ComparableReports reports whether baseline and candidate invocations agree
// on every comparability-relevant setting (everything except the model, the
// KLD logits path values, and the candidate-only --kl-divergence switch).
// Equivalent current and legacy flag aliases compare by their setting
// semantics, not literal spelling.
func ComparableReports(baseline, candidate Invocation) bool {
	norm := func(iv Invocation) []string {
		aliasToName := map[string]string{kldBaseFlag: "kld-base", kldFlag: "kld"}
		for _, alias := range modelAliases {
			aliasToName[alias] = "model"
		}
		for _, spec := range evalFlagSpecs() {
			for _, alias := range spec.aliases {
				aliasToName[alias] = spec.name
			}
		}
		var out []string
		for i := 0; i < len(iv.Argv); i++ {
			a := iv.Argv[i]
			name, known := aliasToName[a]
			if !known {
				out = append(out, a)
				continue
			}
			if name == "kld" {
				// Candidate-only boolean; expected on candidate, forbidden on
				// baseline, and not a comparability setting.
				continue
			}
			if i+1 >= len(iv.Argv) {
				out = append(out, name, "<missing>")
				continue
			}
			i++
			if name == "model" {
				continue
			}
			if name == "kld-base" {
				out = append(out, name)
				continue
			}
			out = append(out, name, iv.Argv[i])
		}
		return out
	}
	b, c := norm(baseline), norm(candidate)
	if len(b) != len(c) {
		return false
	}
	for i := range b {
		if b[i] != c[i] {
			return false
		}
	}
	return true
}

// EvalMetrics aggregates every signal parsed from one llama-perplexity run.
type EvalMetrics struct {
	Perplexity float64 `json:"perplexity"`
	PPLStdErr  float64 `json:"pplStdErr,omitempty"`
	HasPPL     bool    `json:"hasPPL"`
	MeanKLD    float64 `json:"meanKLD,omitempty"`
	HasMeanKLD bool    `json:"hasMeanKLD"`
	P95KLD     float64 `json:"p95KLD,omitempty"`
	HasP95     bool    `json:"hasP95"`
	MaxKLD     float64 `json:"maxKLD,omitempty"`
	HasMax     bool    `json:"hasMax"`
	HasKLD     bool    `json:"hasKLD"`
	RMSDeltaP  float64 `json:"rmsDeltaP,omitempty"`
	HasRMS     bool    `json:"hasRMS"`
	// SameTop is the fraction of tokens whose argmax matches the baseline
	// (llama-perplexity "Same top p"). It is the Divergence-300 @32
	// trajectory-match proxy when only next-token agreement is available:
	// prefix / "same top p1" / first-32 token lines are preferred when the
	// tool prints them; otherwise this is the corpus-wide same-top.
	SameTop    float64 `json:"sameTop,omitempty"`
	HasSameTop bool    `json:"hasSameTop"`
	// CVaRKLD is mean of the worst 5% of parsed per-chunk/token KLD samples.
	// HasCVaR is false unless a sample list was parsed; p95 is never used as a stand-in.
	CVaRKLD float64 `json:"cvarKLD,omitempty"`
	HasCVaR bool    `json:"hasCVaR"`
}

var (
	pplCurrentRe = regexp.MustCompile(`(?i)Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*\+/-\s*([0-9]+(?:\.[0-9]+)?)`)
	pplLegacyRe  = regexp.MustCompile(`(?i)(?:perplexity|ppl)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)`)
	kldMeanRe    = regexp.MustCompile(`(?i)\bmean\s+KLD\s*[:=]\s*([0-9.eE+-]+)`)
	kldP95Re     = regexp.MustCompile(`(?i)(?:\bp95\s+KLD|\b95(?:\.0+)?%\s+KLD)\s*[:=]\s*([0-9.eE+-]+)`)
	kldMaxRe     = regexp.MustCompile(`(?i)\bmax(?:imum)?\s+KLD\s*[:=]\s*([0-9.eE+-]+)`)
	rmsDeltaPRe  = regexp.MustCompile(`(?i)rms\s*(?:delta|Δ)\s*p\s*[:=]\s*([0-9.eE+-]+)`)
	// Corpus-wide "Same top p: 97.54%" / "Same top: 97.54%" (optional ±).
	// Does not match "Same top p1" / "Same top p32" (see sameTopPrefixRe).
	sameTopRe = regexp.MustCompile(`(?i)same\s*top(?:\s*p)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:(?:\+/-|±)\s*[0-9.]+)?\s*%`)
	// Prefix-horizon labels such as "Same top p1: 90%" or "Same top p32: 88%".
	sameTopPrefixRe = regexp.MustCompile(`(?i)same\s*top\s*p(\d+)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:(?:\+/-|±)\s*[0-9.]+)?\s*%`)
	chunkKLDRe      = regexp.MustCompile(`(?i)chunk\s+\d+[:\s].*?\b(?:kld|kl-div(?:ergence)?)\b[:=\s]+([0-9.eE+-]+)`)
	tokenKLDRe      = regexp.MustCompile(`(?i)\btoken\s+\d+.*?\bkld[:=\s]+([0-9.eE+-]+)`)
	kldIndexRe      = regexp.MustCompile(`(?i)\bkld\[\d+\]\s*[:=]\s*([0-9.eE+-]+)`)
	// Per-token argmax agreement, e.g. "token 12: kld=0.01 same_top=1".
	tokenSameTopRe = regexp.MustCompile(`(?i)\btoken\s+(\d+)\b[^\n]*?\b(?:same[_\s-]*top|top-?1[_\s-]*(?:agree(?:ment)?|match)|argmax[_\s-]*match)\s*[:=]\s*(1|0|true|false|yes|no)\b`)
)

// divHorizonTokens is the Unsloth Divergence-300 @32 greedy length we can
// approximate from teacher-forced next-token same-top lines.
const divHorizonTokens = 32

func parseFloat(s string) (float64, bool) {
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	return v, err == nil
}

// ParseEvalMetrics extracts perplexity (current "Final estimate" and legacy
// "perplexity =" formats), KLD mean/p95/max, RMS delta-p, and same-top
// metrics from llama-perplexity output. Same-top prefers a Divergence-300
// @32 prefix (token lines or "same top pN") when printed; otherwise the
// corpus-wide "Same top p". At least one signal must parse.
func ParseEvalMetrics(output string) (EvalMetrics, error) {
	var m EvalMetrics
	if s := pplCurrentRe.FindStringSubmatch(output); s != nil {
		if v, ok := parseFloat(s[1]); ok {
			m.Perplexity, m.HasPPL = v, true
		}
		if v, ok := parseFloat(s[2]); ok {
			m.PPLStdErr = v
		}
	} else if s := pplLegacyRe.FindStringSubmatch(output); s != nil {
		if v, ok := parseFloat(s[1]); ok {
			m.Perplexity, m.HasPPL = v, true
		}
	}
	if s := kldMeanRe.FindStringSubmatch(output); s != nil {
		if v, ok := parseFloat(s[1]); ok {
			m.MeanKLD, m.HasMeanKLD, m.HasKLD = v, true, true
		}
	}
	if s := kldP95Re.FindStringSubmatch(output); s != nil {
		if v, ok := parseFloat(s[1]); ok {
			m.P95KLD, m.HasP95, m.HasKLD = v, true, true
		}
	}
	if s := kldMaxRe.FindStringSubmatch(output); s != nil {
		if v, ok := parseFloat(s[1]); ok {
			m.MaxKLD, m.HasMax, m.HasKLD = v, true, true
		}
	}
	if s := rmsDeltaPRe.FindStringSubmatch(output); s != nil {
		if v, ok := parseFloat(s[1]); ok {
			m.RMSDeltaP, m.HasRMS = v, true
		}
	}
	parseSameTop(output, &m)
	if samples := parseKLDSamples(output); len(samples) >= 2 {
		m.CVaRKLD, m.HasCVaR, m.HasKLD = cvarTail(samples, 0.05), true, true
	}
	if !m.HasPPL && !m.HasKLD && !m.HasRMS && !m.HasSameTop {
		return EvalMetrics{}, fmt.Errorf("orchestrate: no recognizable metrics in output")
	}
	return m, nil
}

func parseSameTop(output string, m *EvalMetrics) {
	global, hasGlobal := parseGlobalSameTop(output)
	prefixN, hasPrefixN := parsePrefixSameTop(output)
	tokenPrefix, nTok := parseTokenSameTopPrefix(output)

	// Prefer a horizon-limited estimate (first 32 next-token agreements, or
	// "same top p32"/"p1") as the Divergence-300 @32 stand-in. Fall back to
	// the corpus-wide Same top p llama-perplexity always prints.
	switch {
	case nTok >= 2 || (nTok >= 1 && !hasPrefixN && !hasGlobal):
		m.SameTop, m.HasSameTop = tokenPrefix, true
	case hasPrefixN:
		m.SameTop, m.HasSameTop = prefixN, true
	case hasGlobal:
		m.SameTop, m.HasSameTop = global, true
	}
}

func parseGlobalSameTop(output string) (float64, bool) {
	matches := sameTopRe.FindAllStringSubmatch(output, -1)
	if len(matches) == 0 {
		return 0, false
	}
	s := matches[len(matches)-1]
	v, ok := parseFloat(s[1])
	if !ok {
		return 0, false
	}
	return percentToUnit(v), true
}

func parsePrefixSameTop(output string) (float64, bool) {
	bestN := 0
	var best float64
	for _, s := range sameTopPrefixRe.FindAllStringSubmatch(output, -1) {
		n, err := strconv.Atoi(s[1])
		if err != nil || n < 1 || n > divHorizonTokens {
			continue
		}
		v, ok := parseFloat(s[2])
		if !ok {
			continue
		}
		if n >= bestN {
			bestN, best = n, percentToUnit(v)
		}
	}
	if bestN == 0 {
		return 0, false
	}
	return best, true
}

type tokenAgree struct {
	idx int
	v   float64
}

func parseTokenSameTopPrefix(output string) (float64, int) {
	byIdx := make(map[int]float64)
	for _, s := range tokenSameTopRe.FindAllStringSubmatch(output, -1) {
		idx, err := strconv.Atoi(s[1])
		if err != nil {
			continue
		}
		v, ok := parseAgreeFlag(s[2])
		if !ok {
			continue
		}
		byIdx[idx] = v
	}
	if len(byIdx) == 0 {
		return 0, 0
	}
	toks := make([]tokenAgree, 0, len(byIdx))
	for idx, v := range byIdx {
		toks = append(toks, tokenAgree{idx: idx, v: v})
	}
	sort.Slice(toks, func(i, j int) bool { return toks[i].idx < toks[j].idx })
	n := len(toks)
	if n > divHorizonTokens {
		n = divHorizonTokens
	}
	var sum float64
	for _, t := range toks[:n] {
		sum += t.v
	}
	return sum / float64(n), n
}

func parseAgreeFlag(s string) (float64, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "1", "true", "yes":
		return 1, true
	case "0", "false", "no":
		return 0, true
	}
	return 0, false
}

func percentToUnit(v float64) float64 {
	if v > 1 {
		return v / 100
	}
	return v
}

func parseKLDSamples(output string) []float64 {
	var out []float64
	add := func(re *regexp.Regexp) {
		for _, s := range re.FindAllStringSubmatch(output, -1) {
			if v, ok := parseFloat(s[1]); ok && v >= 0 && !math.IsNaN(v) && !math.IsInf(v, 0) {
				out = append(out, v)
			}
		}
	}
	add(chunkKLDRe)
	add(tokenKLDRe)
	add(kldIndexRe)
	return out
}

// cvarTail is the mean of the worst alpha fraction (at least one sample).
func cvarTail(samples []float64, alpha float64) float64 {
	if len(samples) == 0 {
		return 0
	}
	sorted := append([]float64(nil), samples...)
	sort.Float64s(sorted)
	n := len(sorted)
	k := int(math.Ceil(float64(n) * alpha))
	if k < 1 {
		k = 1
	}
	if k > n {
		k = n
	}
	var s float64
	for _, v := range sorted[n-k:] {
		s += v
	}
	return s / float64(k)
}
