package quantize

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDetectChatFamily(t *testing.T) {
	cases := []struct {
		arch, tmpl string
		want       chatFamily
	}{
		{"llama", "<|im_start|>{{ role }}", chatFamilyChatML},
		{"llama", "{% set x %}<|start_header_id|>user", chatFamilyLlama3},
		{"gemma2", "<start_of_turn>user", chatFamilyGemma},
		{"mistral", "[INST] {{ content }} [/INST]", chatFamilyMistral},
		{"phi3", "<|user|>\n{{ content }}<|end|>\n<|assistant|>", chatFamilyPhi3},
		{"muse-glimmer", "<|start|>user<|message|>{{ content }}<|eot|>", chatFamilyStartMessage},
		{"muse-glimmer", "", chatFamilyStartMessage},
		{"llama", "", chatFamilyPlain},
	}
	for _, c := range cases {
		if got := detectChatFamily(c.arch, c.tmpl); got != c.want {
			t.Errorf("arch=%q tmpl=%q: got %s want %s", c.arch, c.tmpl[:min(40, len(c.tmpl))], got, c.want)
		}
	}
}

func TestWrapCalibrationFamilies(t *testing.T) {
	src := "User: What is 2^10?\nAssistant: 1024.\nThe river cut limestone.\n"
	glimmer := wrapCalibration(chatFamilyStartMessage, src)
	for _, tok := range []string{"<|start|>user<|message|>", "<|start|>assistant to=user<|message|>", "<|eot|>"} {
		if !strings.Contains(glimmer, tok) {
			t.Errorf("glimmer wrap missing %q in %q", tok, glimmer)
		}
	}
	if strings.Contains(glimmer, "Prime Directive") || strings.Contains(glimmer, "default_persona") {
		t.Fatal("must not inject the Glimmer system persona into calibration")
	}
	if !strings.Contains(glimmer, "What is 2^10?") || !strings.Contains(glimmer, "The river cut limestone.") {
		t.Fatalf("lost content: %q", glimmer)
	}

	chatml := wrapCalibration(chatFamilyChatML, src)
	if !strings.Contains(chatml, "<|im_start|>user\nWhat is 2^10?<|im_end|>") {
		t.Fatalf("chatml: %q", chatml)
	}
	llama3 := wrapCalibration(chatFamilyLlama3, src)
	if !strings.Contains(llama3, "<|start_header_id|>user<|end_header_id|>") || !strings.Contains(llama3, "<|eot_id|>") {
		t.Fatalf("llama3: %q", llama3)
	}

	again := wrapCalibration(chatFamilyStartMessage, glimmer)
	if again != glimmer {
		t.Fatal("wrapping twice should be a no-op")
	}
	plain := wrapCalibration(chatFamilyPlain, src)
	if !strings.Contains(plain, "User: What is 2^10?") {
		t.Fatalf("plain should keep User/Assistant labels: %q", plain)
	}
}

func TestSplitBundledCalibrationDomains(t *testing.T) {
	text := stripCalHeader(string(defaultCalibrationBytes()))
	domains := splitCalibrationDomains(text)
	got := map[string]int{}
	for _, d := range domains {
		got[d.Name] = len(d.Text)
		if strings.TrimSpace(d.Text) == "" {
			t.Fatalf("empty domain %s", d.Name)
		}
	}
	for _, name := range []string{"prose", "facts", "code"} {
		if got[name] < minCalDomainBytes {
			t.Errorf("%s too small: %d bytes (min %d); domains=%v", name, got[name], minCalDomainBytes, got)
		}
	}
	if !strings.Contains(domainText(domains, "code"), "func ") && !strings.Contains(domainText(domains, "code"), "SELECT") {
		t.Fatalf("code domain missing code:\n%s", domainText(domains, "code")[:min(400, len(domainText(domains, "code")))])
	}
	if !strings.Contains(domainText(domains, "facts"), "1928") && !strings.Contains(domainText(domains, "facts"), "Bind these facts") {
		t.Fatalf("facts domain missing binding block:\n%s", domainText(domains, "facts")[:min(400, len(domainText(domains, "facts")))])
	}
}

func domainText(domains []preparedDomain, name string) string {
	for _, d := range domains {
		if d.Name == name {
			return d.Text
		}
	}
	return ""
}

func TestWritePreparedCalibrationWrapsAndSplits(t *testing.T) {
	dir := t.TempDir()
	files, err := writePreparedCalibration(dir, string(defaultCalibrationBytes()), "muse-glimmer",
		"<|start|>user<|message|>x<|eot|>", true)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 3 {
		t.Fatalf("files=%d %+v", len(files), files)
	}
	for _, f := range files {
		b, err := os.ReadFile(f.Path)
		if err != nil {
			t.Fatal(err)
		}
		s := string(b)
		if !strings.Contains(s, "<|start|>user<|message|>") {
			t.Errorf("%s not wrapped: %s", f.Domain, s[:min(200, len(s))])
		}
		if filepath.Base(f.Path) != f.Domain+".txt" {
			t.Errorf("path %s domain %s", f.Path, f.Domain)
		}
	}
	custom, err := writePreparedCalibration(dir, "User: hi\nAssistant: hello\n", "llama", "<|im_start|>", false)
	if err != nil {
		t.Fatal(err)
	}
	if len(custom) != 1 || custom[0].Domain != "all" {
		t.Fatalf("custom should be one file: %+v", custom)
	}
}

func TestPreparedCalLabel(t *testing.T) {
	files := []preparedCal{
		{Domain: "prose", Family: chatFamilyStartMessage},
		{Domain: "facts", Family: chatFamilyStartMessage},
		{Domain: "code", Family: chatFamilyStartMessage},
	}
	got := preparedCalLabel("research", files)
	if !strings.Contains(got, "research") || !strings.Contains(got, "+chat") || !strings.Contains(got, "prose+facts+code") {
		t.Fatalf("label=%q", got)
	}
}

func TestReadCalibrationLimitedRejectsSparseOversizeFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "too-large.txt")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(maxCalibrationFileBytes + 1); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	var total int64
	_, err = readCalibrationLimited(path, &total)
	if err == nil || !strings.Contains(err.Error(), "maximum") {
		t.Fatalf("oversize calibration error = %v", err)
	}
}

func TestFactCalibrationText(t *testing.T) {
	s := factCalibrationText()
	if len(s) < 8000 {
		t.Fatalf("fact cards too short: %d", len(s))
	}
	for _, needle := range []string{"1969", "Apollo 11", "Sputnik", "Gagarin", "Bind these facts"} {
		if !strings.Contains(s, needle) {
			t.Fatalf("missing %q", needle)
		}
	}
	all := defaultCalibrationBytes()
	if len(all) <= len(s) {
		t.Fatal("bundled calibration should include default.txt plus fact cards")
	}
}

func TestDefaultCalibrationIsDeterministicAndLarge(t *testing.T) {
	first := defaultCalibrationBytes()
	second := defaultCalibrationBytes()
	if !bytes.Equal(first, second) {
		t.Fatal("default calibration composition is not deterministic")
	}
	if got := len(strings.Fields(string(first))); got < 100000 {
		t.Fatalf("default calibration has %d unique source tokens, want at least 100000", got)
	}
	if got := len(bytes.TrimSpace(first)) / 5; got < 100000 {
		t.Fatalf("conservative token estimate is %d, want at least 100000", got)
	}
}

func TestCalibrationLexicalDiversity(t *testing.T) {
	fields := strings.Fields(strings.ToLower(string(defaultCalibrationBytes())))
	types := make(map[string]struct{}, len(fields)/4)
	grams := make(map[string]struct{}, len(fields))
	for _, w := range fields {
		types[w] = struct{}{}
	}
	for i := 0; i+5 <= len(fields); i++ {
		grams[strings.Join(fields[i:i+5], " ")] = struct{}{}
	}
	// Calibration-only bytes sit under the full mix; floors still catch a
	// collapse back to the short looping template (~38k types, ~328k 5-grams).
	if len(types) < 30000 {
		t.Fatalf("unique tokens = %d, want >= 30000 (fields=%d)", len(types), len(fields))
	}
	if len(grams) < 400000 {
		t.Fatalf("unique 5-grams = %d, want >= 400000 (types=%d fields=%d)", len(grams), len(types), len(fields))
	}
}

func TestCalibrationPartitionsAreDisjointAndDiverse(t *testing.T) {
	corpus := bundledCalibrationCorpus()
	if len(corpus.Calibration) == 0 || len(corpus.Search) == 0 || len(corpus.Validation) == 0 {
		t.Fatalf("empty partition: calibration=%d search=%d validation=%d", len(corpus.Calibration), len(corpus.Search), len(corpus.Validation))
	}

	ids := map[string]calibrationPartition{}
	texts := map[string]string{}
	domainsByPartition := map[calibrationPartition]map[string]bool{}
	for _, partition := range [][]calibrationRecord{corpus.Calibration, corpus.Search, corpus.Validation} {
		for _, r := range partition {
			if previous, ok := ids[r.ID]; ok {
				t.Fatalf("record %q occurs in %s and %s", r.ID, previous, r.Partition)
			}
			ids[r.ID] = r.Partition
			text := strings.TrimSpace(r.Text)
			if previous, ok := texts[text]; ok {
				t.Fatalf("records %q and %q have identical content", previous, r.ID)
			}
			texts[text] = r.ID
			if r.Source == "" {
				t.Fatalf("record %q has no provenance", r.ID)
			}
			if domainsByPartition[r.Partition] == nil {
				domainsByPartition[r.Partition] = map[string]bool{}
			}
			domainsByPartition[r.Partition][r.Domain] = true
		}
	}
	for _, partition := range []calibrationPartition{partitionCalibration, partitionSearch, partitionValidation} {
		for _, domain := range syntheticCalibrationDomains {
			if !domainsByPartition[partition][domain] {
				t.Fatalf("%s partition lacks %s records", partition, domain)
			}
		}
	}

	calibration := string(defaultCalibrationBytes())
	for _, r := range append(append([]calibrationRecord{}, corpus.Search...), corpus.Validation...) {
		if strings.Contains(calibration, strings.TrimSpace(r.Text)) {
			t.Fatalf("holdout record %q leaked into calibration", r.ID)
		}
	}
}

func TestCustomCalibrationBuildsDisjointUserHoldouts(t *testing.T) {
	var paragraphs []string
	for i := 0; i < 30; i++ {
		paragraphs = append(paragraphs, fmt.Sprintf("User record %02d contains unique calibration text and marker custom-%02d.", i, i))
	}
	corpus, err := customCalibrationCorpus(strings.Join(paragraphs, "\n\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(corpus.Calibration) == 0 || len(corpus.Search) == 0 || len(corpus.Validation) == 0 {
		t.Fatalf("empty custom partition: %+v", corpus)
	}
	seen := map[string]calibrationPartition{}
	for _, records := range [][]calibrationRecord{corpus.Calibration, corpus.Search, corpus.Validation} {
		for _, record := range records {
			if previous, ok := seen[record.ID]; ok {
				t.Fatalf("custom record %s leaked from %s into %s", record.ID, previous, record.Partition)
			}
			seen[record.ID] = record.Partition
			if record.Source != "user-provided calibration corpus" || !strings.Contains(record.Text, "custom-") {
				t.Fatalf("holdout did not come from user corpus: %+v", record)
			}
		}
	}
}

func TestCalibrationPartitionsWriteDomainValidationHoldouts(t *testing.T) {
	dir := t.TempDir()
	if _, err := writeCalibrationPartitions(dir, chatFamilyPlain, bundledCalibrationCorpus()); err != nil {
		t.Fatal(err)
	}
	for _, domain := range syntheticCalibrationDomains {
		for _, part := range []string{"validation", "search"} {
			name := part + "-" + domain + ".txt"
			path := filepath.Join(dir, name)
			st, err := os.Stat(path)
			if err != nil {
				t.Fatalf("domain holdout %s missing: %v", name, err)
			}
			if st.Size() < int64(minDomainHoldoutBytes) {
				t.Fatalf("domain holdout %s is %d bytes; want >= %d (llama-perplexity ctx=4096 floor)", name, st.Size(), minDomainHoldoutBytes)
			}
		}
	}
}

func TestTopUpDomainHoldoutsMovesWithoutDuplication(t *testing.T) {
	var corpus calibrationCorpus
	for i := 0; i < 500; i++ {
		rec := calibrationRecord{
			ID:        fmt.Sprintf("topup-%04d", i),
			Domain:    "prose",
			Source:    "top-up unit test",
			Text:      fmt.Sprintf("User: unique-topup-record-%04d\nAssistant: filler %s ack-%04d.\n", i, strings.Repeat("x", 220), i),
			Partition: partitionCalibration,
		}
		switch {
		case i < 2:
			rec.Partition = partitionSearch
		case i < 4:
			rec.Partition = partitionValidation
		}
		appendCalibrationRecord(&corpus, rec)
	}
	if domainPartitionWrappedBytes(corpus.Search, "prose") >= minDomainHoldoutBytes {
		t.Fatal("precondition: search slice should start under the holdout floor")
	}
	topUpDomainHoldouts(&corpus)
	if got := domainPartitionWrappedBytes(corpus.Search, "prose"); got < minDomainHoldoutBytes {
		t.Fatalf("search still short after top-up: %d", got)
	}
	if got := domainPartitionWrappedBytes(corpus.Validation, "prose"); got < minDomainHoldoutBytes {
		t.Fatalf("validation still short after top-up: %d", got)
	}
	if len(corpus.Calibration) == 0 {
		t.Fatal("top-up emptied calibration")
	}
	seen := map[string]calibrationPartition{}
	for _, records := range [][]calibrationRecord{corpus.Calibration, corpus.Search, corpus.Validation} {
		for _, rec := range records {
			if previous, ok := seen[rec.ID]; ok {
				t.Fatalf("record %s duplicated in %s and %s", rec.ID, previous, rec.Partition)
			}
			seen[rec.ID] = rec.Partition
		}
	}
}

func TestCalibrationActivationCoverage(t *testing.T) {
	corpus := bundledCalibrationCorpus()
	all := append(append(append([]calibrationRecord{}, corpus.Calibration...), corpus.Search...), corpus.Validation...)
	var multilingual, structured, longContext, directAnswer, privacyBoundary, chatMulti bool
	for _, r := range all {
		switch r.Domain {
		case "multilingual":
			if strings.Contains(r.Text, "測定値") && strings.Contains(r.Text, "単位") {
				multilingual = true
			}
		case "structured-tool":
			if strings.Contains(r.Text, "<tool_call>") && strings.Contains(r.Text, `"arguments"`) && strings.Contains(r.Text, "<tool_result>") {
				structured = true
			}
		case "long-context":
			if len(r.Text) >= 2000 && strings.Contains(r.Text, "first and final entries") {
				longContext = true
			}
		case "refusal-adjacent":
			if strings.Contains(r.Text, "Refusal-adjacent benign request") && strings.Contains(r.Text, "Assistant: Direct answer:") {
				directAnswer = true
			}
			if strings.Contains(r.Text, "private person's home address") && strings.Contains(r.Text, "I cannot help identify or expose") {
				privacyBoundary = true
			}
		case "chat":
			if strings.Count(r.Text, "User:") >= 3 {
				chatMulti = true
			}
		}
	}
	if !multilingual || !structured || !longContext || !directAnswer || !privacyBoundary || !chatMulti {
		t.Fatalf("coverage multilingual=%v structured=%v long_context=%v direct_answer=%v privacy_boundary=%v chat=%v",
			multilingual, structured, longContext, directAnswer, privacyBoundary, chatMulti)
	}
}

func TestWritePreparedCalibrationWritesHoldoutsAndProvenance(t *testing.T) {
	dir := t.TempDir()
	files, err := writePreparedCalibration(dir, string(defaultCalibrationBytes()), "llama", "<|im_start|>", true)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 || files[0].ProvenancePath == "" {
		t.Fatalf("prepared files lack provenance: %+v", files)
	}
	for _, name := range []string{"search.txt", "validation.txt"} {
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		text := string(raw)
		if !strings.Contains(text, "<|im_start|>user") {
			t.Fatalf("%s was not chat wrapped", name)
		}
		if wrapCalibration(chatFamilyChatML, text) != text {
			t.Fatalf("%s is not idempotently wrapped", name)
		}
	}

	raw, err := os.ReadFile(files[0].ProvenancePath)
	if err != nil {
		t.Fatal(err)
	}
	var manifest calibrationProvenance
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	corpus := bundledCalibrationCorpus()
	want := len(corpus.Calibration) + len(corpus.Search) + len(corpus.Validation)
	if manifest.Version != 1 || len(manifest.Records) != want {
		t.Fatalf("provenance version=%d records=%d want=%d", manifest.Version, len(manifest.Records), want)
	}
	for _, r := range manifest.Records {
		if r.ID == "" || r.Source == "" || len(r.SHA256) != 64 || r.Bytes <= 0 {
			t.Fatalf("incomplete provenance record: %+v", r)
		}
	}
}

func TestCalibrationRecordsInterleaveDomains(t *testing.T) {
	var clumped []calibrationRecord
	for i := 0; i < 30; i++ {
		clumped = append(clumped, calibrationRecord{Domain: "prose", ID: fmt.Sprintf("p%d", i), Text: "p"})
	}
	for i := 0; i < 30; i++ {
		clumped = append(clumped, calibrationRecord{Domain: "code", ID: fmt.Sprintf("c%d", i), Text: "c"})
	}
	for i := 0; i < 30; i++ {
		clumped = append(clumped, calibrationRecord{Domain: "facts", ID: fmt.Sprintf("f%d", i), Text: "f"})
	}
	if run := longestDomainRun(clumped); run != 30 {
		t.Fatalf("precondition: clump run = %d, want 30", run)
	}
	got := interleaveRecordsByDomain(clumped)
	if run := longestDomainRun(got); run != 1 {
		t.Fatalf("interleaved run = %d, want 1", run)
	}
	if len(got) != len(clumped) {
		t.Fatalf("interleave dropped records: %d vs %d", len(got), len(clumped))
	}

	corpus := bundledCalibrationCorpus()
	if run := longestDomainRun(corpus.Calibration); run > 32 {
		t.Fatalf("bundled calibration domain run = %d is still a long clump (n=%d)", run, len(corpus.Calibration))
	}
}

func longestDomainRun(records []calibrationRecord) int {
	max, cur := 0, 0
	prev := ""
	for _, r := range records {
		d := r.Domain
		if d == "" {
			d = "_"
		}
		if d == prev {
			cur++
		} else {
			cur = 1
			prev = d
		}
		if cur > max {
			max = cur
		}
	}
	return max
}
