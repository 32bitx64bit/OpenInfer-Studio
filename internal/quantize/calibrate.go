package quantize

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/internal/models"
)

// Chat-shaped calibration so llama-imatrix --parse-special sees the same
// special tokens Chat uses. The bundled corpus is written as one mixed
// file with records round-robined by domain. Custom calibration_path is
// wrapped as a single file.

type chatFamily string

const (
	chatFamilyPlain        chatFamily = "plain"
	chatFamilyChatML       chatFamily = "chatml"
	chatFamilyLlama3       chatFamily = "llama3"
	chatFamilyStartMessage chatFamily = "start_message" // Harmony / Muse Glimmer
	chatFamilyGemma        chatFamily = "gemma"
	chatFamilyMistral      chatFamily = "mistral"
	chatFamilyPhi3         chatFamily = "phi3"
)

const minCalDomainBytes = 5120 // ~1024 tokens at 5 chars; 512 ctx needs two windows

// minDomainHoldoutBytes is llama-perplexity's skip floor at ctx=4096
// (quantlab/pipeline.minPerplexityCorpusBytes = ctx*2*4). Search and
// validation domain files written by writeCalibrationPartitions must meet it.
const minDomainHoldoutBytes = 32768

const (
	maxCalibrationFileBytes  int64 = 64 << 20
	maxCalibrationTotalBytes int64 = 256 << 20
)

type calTurn struct {
	Role    string // user, assistant, system, text
	Content string
}

type preparedCal struct {
	Domain         string
	Path           string
	Family         chatFamily
	ProvenancePath string
}

type preparedDomain struct {
	Name string
	Text string
}

type calibrationProvenance struct {
	Version   int                       `json:"version"`
	Partition string                    `json:"partition_algorithm"`
	Records   []calibrationRecordSource `json:"records"`
}

type calibrationRecordSource struct {
	ID        string `json:"id"`
	Partition string `json:"partition"`
	Domain    string `json:"domain"`
	Source    string `json:"source"`
	SHA256    string `json:"sha256"`
	Bytes     int    `json:"bytes"`
}

var (
	calCodeLineRe = regexp.MustCompile(`(?m)^\s*(func\s|fn\s|def\s|package\s|type\s|var\s\(|const\s|SELECT\s|INSERT\s|UPDATE\s|assert\s|impl\s|#include\s|class\s|struct\s|enum\s)`)
	calFactRe     = regexp.MustCompile(`(?i)(?:\b(?:1[4-9]\d{2}|20[0-2]\d)\b|Z\s*=\s*\d+|2\^\d+|\d{4}-\d{2}-\d{2}|\d+(?:\.\d+)?\s*(?:km|kg|kPa|GiB|MiB|cm\b|m/s|°C)|Bind these facts|Chemistry cards|factual-binding|ISO dates|seconds in an hour|glued to)`)
)

func detectChatFamily(arch, tmpl string) chatFamily {
	t := strings.ToLower(tmpl)
	switch {
	case strings.Contains(tmpl, "<|start|>") && strings.Contains(tmpl, "<|message|>"):
		return chatFamilyStartMessage
	case strings.Contains(tmpl, "<|im_start|>") || strings.Contains(t, "im_start"):
		return chatFamilyChatML
	case strings.Contains(tmpl, "<|start_header_id|>"):
		return chatFamilyLlama3
	case strings.Contains(tmpl, "<start_of_turn>"):
		return chatFamilyGemma
	case strings.Contains(tmpl, "[INST]"):
		return chatFamilyMistral
	case strings.Contains(tmpl, "<|user|>") && strings.Contains(tmpl, "<|assistant|>"):
		return chatFamilyPhi3
	case gguf.IsMuseGlimmerChat(arch):
		return chatFamilyStartMessage
	default:
		return chatFamilyPlain
	}
}

func alreadyWrapped(family chatFamily, text string) bool {
	switch family {
	case chatFamilyStartMessage:
		return strings.Contains(text, "<|start|>user") || strings.Contains(text, "<|message|>")
	case chatFamilyChatML:
		return strings.Contains(text, "<|im_start|>")
	case chatFamilyLlama3:
		return strings.Contains(text, "<|start_header_id|>")
	case chatFamilyGemma:
		return strings.Contains(text, "<start_of_turn>")
	case chatFamilyMistral:
		return strings.Contains(text, "[INST]")
	case chatFamilyPhi3:
		return strings.Contains(text, "<|user|>") && strings.Contains(text, "<|end|>")
	default:
		return false
	}
}

func stripCalHeader(text string) string {
	headN := 800
	if len(text) < headN {
		headN = len(text)
	}
	head := text[:headN]
	if !strings.Contains(head, "imatrix calibration") && !strings.Contains(head, "OpenInfer Studio default") {
		return text
	}
	if i := strings.Index(text, "\nUser:"); i >= 0 {
		return strings.TrimLeft(text[i+1:], "\n")
	}
	return text
}

func cutCalRole(line string) (role, rest string, ok bool) {
	for _, p := range []struct {
		prefix, role string
	}{
		{"User:", "user"},
		{"Assistant:", "assistant"},
		{"System:", "system"},
	} {
		if strings.HasPrefix(line, p.prefix) {
			return p.role, strings.TrimSpace(strings.TrimPrefix(line, p.prefix)), true
		}
	}
	return "", "", false
}

func parseCalTurns(text string) []calTurn {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	lines := strings.Split(text, "\n")
	var turns []calTurn
	var cur calTurn
	flush := func() {
		s := strings.TrimSpace(cur.Content)
		if s == "" {
			cur = calTurn{}
			return
		}
		cur.Content = s
		turns = append(turns, cur)
		cur = calTurn{}
	}
	for _, line := range lines {
		if role, rest, ok := cutCalRole(line); ok {
			flush()
			cur = calTurn{Role: role, Content: rest}
			continue
		}
		if cur.Role == "" {
			cur.Role = "text"
		}
		if cur.Content != "" {
			cur.Content += "\n"
		}
		cur.Content += line
	}
	flush()
	return turns
}

func formatCalTurn(t calTurn) string {
	switch t.Role {
	case "user":
		return "User: " + t.Content
	case "assistant":
		return "Assistant: " + t.Content
	case "system":
		return "System: " + t.Content
	default:
		return t.Content
	}
}

func groupCalBlocks(text string) []string {
	turns := parseCalTurns(text)
	var blocks []string
	for i := 0; i < len(turns); i++ {
		if i+1 < len(turns) && turns[i].Role == "user" && turns[i+1].Role == "assistant" {
			blocks = append(blocks, formatCalTurn(turns[i])+"\n"+formatCalTurn(turns[i+1]))
			i++
			continue
		}
		blocks = append(blocks, formatCalTurn(turns[i]))
	}
	return blocks
}

func classifyCalBlock(s string) string {
	if strings.Contains(s, "factual-binding") || strings.Contains(s, "Bind these facts") ||
		strings.Contains(s, "Chemistry cards") || strings.Contains(s, "Math cards") ||
		strings.Contains(s, "Medicine-history") || strings.Contains(s, "Measurement card") {
		return "facts"
	}
	codeN := len(calCodeLineRe.FindAllString(s, -1))
	factN := len(calFactRe.FindAllString(s, -1))
	if codeN >= 2 && codeN >= factN {
		return "code"
	}
	if factN >= 3 {
		return "facts"
	}
	if codeN >= 1 && factN < 2 {
		return "code"
	}
	return "prose"
}

func splitCalibrationDomains(text string) []preparedDomain {
	buckets := map[string]*strings.Builder{
		"prose": {},
		"facts": {},
		"code":  {},
	}
	for _, block := range groupCalBlocks(text) {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		d := classifyCalBlock(block)
		sb := buckets[d]
		if sb.Len() > 0 {
			sb.WriteString("\n\n")
		}
		sb.WriteString(block)
	}
	order := []string{"prose", "facts", "code"}
	var out []preparedDomain
	for _, name := range order {
		s := strings.TrimSpace(buckets[name].String())
		if s == "" {
			continue
		}
		out = append(out, preparedDomain{Name: name, Text: s})
	}
	return coalesceDomains(out, minCalDomainBytes)
}

func coalesceDomains(in []preparedDomain, min int) []preparedDomain {
	if len(in) <= 1 {
		return in
	}
	largest := 0
	for i := 1; i < len(in); i++ {
		if len(in[i].Text) > len(in[largest].Text) {
			largest = i
		}
	}
	extra := ""
	var keep []preparedDomain
	for i, d := range in {
		if i != largest && len(d.Text) < min {
			extra += "\n\n" + d.Text
			continue
		}
		keep = append(keep, d)
	}
	if extra != "" {
		for i := range keep {
			if keep[i].Name == in[largest].Name {
				keep[i].Text += extra
			}
		}
	}
	return keep
}

func renderTurn(family chatFamily, t calTurn) string {
	role := t.Role
	if role == "text" || role == "" {
		role = "user"
	}
	content := strings.TrimSpace(t.Content)
	switch family {
	case chatFamilyChatML:
		return "<|im_start|>" + role + "\n" + content + "<|im_end|>\n"
	case chatFamilyLlama3:
		return "<|start_header_id|>" + role + "<|end_header_id|>\n\n" + content + "<|eot_id|>"
	case chatFamilyStartMessage:
		switch role {
		case "assistant":
			return "<|start|>assistant to=user<|message|>" + content + "<|eot|>"
		case "system":
			return "<|start|>system<|message|>" + content + "<|eot|>"
		default:
			return "<|start|>user<|message|>" + content + "<|eot|>"
		}
	case chatFamilyGemma:
		r := role
		if r == "assistant" {
			r = "model"
		}
		return "<start_of_turn>" + r + "\n" + content + "<end_of_turn>\n"
	case chatFamilyMistral:
		if role == "assistant" {
			return content + "</s>"
		}
		return "[INST] " + content + " [/INST] "
	case chatFamilyPhi3:
		return "<|" + role + "|>\n" + content + "<|end|>\n"
	default:
		if role == "assistant" {
			return "Assistant: " + content + "\n"
		}
		if role == "system" {
			return "System: " + content + "\n"
		}
		return "User: " + content + "\n"
	}
}

func wrapCalibration(family chatFamily, text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	if family == chatFamilyPlain || alreadyWrapped(family, text) {
		if !strings.HasSuffix(text, "\n") {
			text += "\n"
		}
		return text
	}
	turns := parseCalTurns(text)
	if len(turns) == 0 {
		return renderTurn(family, calTurn{Role: "user", Content: text}) + "\n"
	}
	var b strings.Builder
	for _, t := range turns {
		b.WriteString(renderTurn(family, t))
	}
	s := b.String()
	if !strings.HasSuffix(s, "\n") {
		s += "\n"
	}
	return s
}

func writePreparedCalibration(dir, text, arch, tmpl string, split bool) ([]preparedCal, error) {
	text = stripCalHeader(text)
	family := detectChatFamily(arch, tmpl)
	var domains []preparedDomain
	if split {
		domains = splitCalibrationDomains(text)
	}
	if len(domains) == 0 {
		domains = []preparedDomain{{Name: "all", Text: text}}
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	provenancePath := ""
	if split {
		var err error
		provenancePath, err = writeCalibrationPartitions(dir, family, bundledCalibrationCorpus())
		if err != nil {
			return nil, err
		}
	}
	var out []preparedCal
	for _, d := range domains {
		wrapped := wrapCalibration(family, d.Text)
		if strings.TrimSpace(wrapped) == "" {
			continue
		}
		path := filepath.Join(dir, d.Name+".txt")
		if err := os.WriteFile(path, []byte(wrapped), 0o644); err != nil {
			return nil, err
		}
		out = append(out, preparedCal{Domain: d.Name, Path: path, Family: family, ProvenancePath: provenancePath})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("calibration text is empty after wrap")
	}
	return out, nil
}

func writeCalibrationPartitions(dir string, family chatFamily, corpus calibrationCorpus) (string, error) {
	for _, p := range []struct {
		name    string
		records []calibrationRecord
	}{
		{name: string(partitionSearch), records: corpus.Search},
		{name: string(partitionValidation), records: corpus.Validation},
	} {
		text := wrapCalibration(family, renderCalibrationRecords(p.records))
		if strings.TrimSpace(text) == "" {
			return "", fmt.Errorf("%s calibration partition is empty", p.name)
		}
		if err := os.WriteFile(filepath.Join(dir, p.name+".txt"), []byte(text), 0o644); err != nil {
			return "", err
		}
		byDomain := make(map[string][]calibrationRecord)
		for _, record := range p.records {
			if record.Domain != "" {
				byDomain[record.Domain] = append(byDomain[record.Domain], record)
			}
		}
		domains := make([]string, 0, len(byDomain))
		for domain := range byDomain {
			domains = append(domains, domain)
		}
		sort.Strings(domains)
		for _, domain := range domains {
			domainText := wrapCalibration(family, renderCalibrationRecords(byDomain[domain]))
			if strings.TrimSpace(domainText) == "" {
				continue
			}
			name := p.name + "-" + domain + ".txt"
			if err := os.WriteFile(filepath.Join(dir, name), []byte(domainText), 0o644); err != nil {
				return "", err
			}
		}
	}

	manifest := calibrationProvenance{
		Version:   1,
		Partition: "generated records use fnv1a32(record_id) mod 10: 0-7 calibration, 8 search, 9 validation; short domain holdouts may reassign extra unused calibration records of the same domain without duplication; project seed records are pinned to calibration",
	}
	all := make([]calibrationRecord, 0, len(corpus.Calibration)+len(corpus.Search)+len(corpus.Validation))
	all = append(all, corpus.Calibration...)
	all = append(all, corpus.Search...)
	all = append(all, corpus.Validation...)
	for _, r := range all {
		sum := sha256.Sum256([]byte(strings.TrimSpace(r.Text)))
		manifest.Records = append(manifest.Records, calibrationRecordSource{
			ID:        r.ID,
			Partition: string(r.Partition),
			Domain:    r.Domain,
			Source:    r.Source,
			SHA256:    hex.EncodeToString(sum[:]),
			Bytes:     len(strings.TrimSpace(r.Text)),
		})
	}
	raw, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, "provenance.json")
	if err := os.WriteFile(path, append(raw, '\n'), 0o644); err != nil {
		return "", err
	}
	return path, nil
}

func concatPrepared(dir string, files []preparedCal) (preparedCal, error) {
	var b strings.Builder
	var total int64
	for i, f := range files {
		raw, err := readCalibrationLimited(f.Path, &total)
		if err != nil {
			return preparedCal{}, err
		}
		if i > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString(raw)
	}
	path := filepath.Join(dir, "all.txt")
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		return preparedCal{}, err
	}
	fam := chatFamilyPlain
	if len(files) > 0 {
		fam = files[0].Family
	}
	provenancePath := ""
	if len(files) > 0 {
		provenancePath = files[0].ProvenancePath
	}
	return preparedCal{Domain: "all", Path: path, Family: fam, ProvenancePath: provenancePath}, nil
}

func (m *Manager) prepareJobCalibration(j *Job, src *models.Model, calPath string) ([]preparedCal, error) {
	var total int64
	raw, err := readCalibrationLimited(calPath, &total)
	if err != nil {
		return nil, err
	}
	var arch, tmpl string
	if src != nil && src.PrimaryPath != "" {
		if md, err := gguf.ParseFile(src.PrimaryPath); err == nil && md != nil {
			arch, tmpl = md.Architecture, md.ChatTemplate
		}
	}
	split := bytes.Equal([]byte(raw), defaultCalibrationBytes())
	dir := filepath.Join(m.layout.QuantJobs, j.ID, "calibration")
	family := detectChatFamily(arch, tmpl)
	if split {
		// Write the interleaved mixed file as the imatrix input. Domain
		// buckets concatenated in prose/facts/code order would clump one
		// domain for a long run. Holdouts still come from the corpus.
		files, err := writePreparedCalibration(dir, raw, arch, tmpl, false)
		if err != nil {
			return nil, err
		}
		provenancePath, err := writeCalibrationPartitions(dir, family, bundledCalibrationCorpus())
		if err != nil {
			return nil, err
		}
		for i := range files {
			files[i].ProvenancePath = provenancePath
			files[i].Domain = "mixed"
		}
		return files, nil
	}
	if usesQuantlab(j.Request) {
		corpus, err := customCalibrationCorpus(raw)
		if err != nil {
			return nil, err
		}
		calibrationText := renderCalibrationRecords(corpus.Calibration)
		files, err := writePreparedCalibration(dir, calibrationText, arch, tmpl, false)
		if err != nil {
			return nil, err
		}
		provenancePath, err := writeCalibrationPartitions(dir, family, corpus)
		if err != nil {
			return nil, err
		}
		for i := range files {
			files[i].ProvenancePath = provenancePath
		}
		return files, nil
	}
	files, err := writePreparedCalibration(dir, raw, arch, tmpl, split)
	if err != nil {
		return nil, err
	}
	return files, nil
}

func customCalibrationCorpus(raw string) (calibrationCorpus, error) {
	text := stripCalHeader(raw)
	var records []string
	for _, paragraph := range strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n\n") {
		if paragraph = strings.TrimSpace(paragraph); paragraph != "" {
			records = append(records, paragraph)
		}
	}
	if len(records) < 3 {
		records = records[:0]
		fields := strings.Fields(text)
		parts := len(fields) / 128
		if parts < 3 {
			parts = 3
		}
		if parts > 64 {
			parts = 64
		}
		for i := 0; i < parts && len(fields) >= parts; i++ {
			start := i * len(fields) / parts
			end := (i + 1) * len(fields) / parts
			if start < end {
				records = append(records, strings.Join(fields[start:end], " "))
			}
		}
	}
	unique := make([]string, 0, len(records))
	seen := map[[32]byte]bool{}
	for _, record := range records {
		sum := sha256.Sum256([]byte(strings.TrimSpace(record)))
		if seen[sum] {
			continue
		}
		seen[sum] = true
		unique = append(unique, record)
	}
	records = unique
	if len(records) < 3 {
		return calibrationCorpus{}, fmt.Errorf("custom Dynamic calibration needs at least three non-empty records (or enough text to split into calibration, search, and validation holdouts)")
	}
	sort.Slice(records, func(i, j int) bool {
		left := sha256.Sum256([]byte(records[i]))
		right := sha256.Sum256([]byte(records[j]))
		return bytes.Compare(left[:], right[:]) < 0
	})
	calibrationCount := len(records) * 8 / 10
	if calibrationCount < 1 {
		calibrationCount = 1
	}
	if calibrationCount > len(records)-2 {
		calibrationCount = len(records) - 2
	}
	searchCount := (len(records) - calibrationCount) / 2
	if searchCount < 1 {
		searchCount = 1
	}
	var corpus calibrationCorpus
	for i, text := range records {
		sum := sha256.Sum256([]byte(text))
		record := calibrationRecord{
			ID:     "custom-" + hex.EncodeToString(sum[:8]),
			Domain: "custom",
			Source: "user-provided calibration corpus",
			Text:   text,
		}
		switch {
		case i < calibrationCount:
			record.Partition = partitionCalibration
		case i < calibrationCount+searchCount:
			record.Partition = partitionSearch
		default:
			record.Partition = partitionValidation
		}
		appendCalibrationRecord(&corpus, record)
	}
	interleaveCalibrationCorpus(&corpus)
	return corpus, nil
}

// readCalibrationLimited streams a user corpus with explicit per-file and
// aggregate caps. Stat catches sparse files without materializing their holes.
func readCalibrationLimited(path string, total *int64) (string, error) {
	st, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if st.IsDir() {
		return "", fmt.Errorf("calibration path is a directory: %s", path)
	}
	if st.Size() > maxCalibrationFileBytes {
		return "", fmt.Errorf("calibration file %s is %.1f MiB; maximum is %d MiB", filepath.Base(path), float64(st.Size())/(1<<20), maxCalibrationFileBytes>>20)
	}
	if *total > maxCalibrationTotalBytes-st.Size() {
		return "", fmt.Errorf("calibration corpus exceeds the %d MiB total limit", maxCalibrationTotalBytes>>20)
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	var b strings.Builder
	b.Grow(int(st.Size()))
	n, err := io.Copy(&b, io.LimitReader(f, maxCalibrationFileBytes+1))
	if err != nil {
		return "", err
	}
	if n > maxCalibrationFileBytes {
		return "", fmt.Errorf("calibration file %s exceeds the %d MiB limit", filepath.Base(path), maxCalibrationFileBytes>>20)
	}
	*total += n
	return b.String(), nil
}

func preparedCalLabel(preset string, files []preparedCal) string {
	label := strings.TrimSpace(preset)
	if label == "" {
		label = "calibration"
	}
	if len(files) == 0 {
		return label
	}
	if files[0].Family != chatFamilyPlain {
		label += "+chat"
	}
	if len(files) > 1 {
		var names []string
		for _, f := range files {
			names = append(names, f.Domain)
		}
		label += "+" + strings.Join(names, "+")
	} else if files[0].Domain == "mixed" {
		label += "+mixed"
	}
	return label
}
