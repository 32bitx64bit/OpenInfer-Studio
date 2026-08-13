package huggingface

import (
	"strings"
	"testing"
)

func TestQuantOf(t *testing.T) {
	cases := map[string]string{
		"model-Q4_K_M.gguf":                            "Q4_K_M",
		"Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf": "Q4_K_XL",
		"some-model.IQ4_XS.gguf":                       "IQ4_XS",
		"weird.name-q8_0.gguf":                         "Q8_0",
		"model-F16.gguf":                               "F16",
		"model.gguf":                                   "",
		"model-Q4_K_M-00001-of-00002.gguf":             "Q4_K_M",
		"Qwen3-8B-UD-Q4_K_XL.gguf":                     "UD-Q4_K_XL",
		"Llama-3.1-8B-Instruct-UD-IQ3_XXS.gguf":        "UD-IQ3_XXS",
		"Qwen3-8B-UD-Q8_K_XL.gguf":                     "UD-Q8_K_XL",
		"UD-Q5_K_XL/weights.gguf":                      "UD-Q5_K_XL",
		// Requantized uploads keep a leftover .f16.gguf fragment before the real quant.
		"Q4_K_M/amd.Instella-MoE-16B-A3B-Think.f16.gguf.Q4_K_M.gguf": "Q4_K_M",
		"Q2_K/amd.Instella-MoE-16B-A3B-Think.f16.gguf.Q2_K.gguf":     "Q2_K",
		"Q8_0/amd.Instella-MoE-16B-A3B-Think.f16.gguf.Q8_0.gguf":     "Q8_0",
		// Quant only in the parent folder.
		"Q5_K_M/model.gguf":   "Q5_K_M",
		"IQ4_XS/weights.gguf": "IQ4_XS",
	}
	for in, want := range cases {
		if got := quantOf(in); got != want {
			t.Errorf("quantOf(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestGroupFilesUnslothDynamic(t *testing.T) {
	files := []FileEntry{
		{Path: "Qwen3-8B-UD-Q4_K_XL.gguf", Size: 100},
		{Path: "Qwen3-8B-UD-Q8_K_XL.gguf", Size: 200},
		{Path: "Qwen3-8B-F16.gguf", Size: 300},
	}
	groups, _, _ := GroupFiles(files)
	if len(groups) != 3 {
		t.Fatalf("want 3 groups, got %d: %v", len(groups), labels(groups))
	}
	byQuant := map[string]FileGroup{}
	for _, g := range groups {
		byQuant[g.Quant] = g
	}
	for _, q := range []string{"UD-Q4_K_XL", "UD-Q8_K_XL", "F16"} {
		g, ok := byQuant[q]
		if !ok {
			t.Errorf("missing group %q in %v", q, labels(groups))
			continue
		}
		if g.Label != q {
			t.Errorf("quant %q label = %q", q, g.Label)
		}
	}
}

func TestGroupFilesInstellaStyleRequant(t *testing.T) {
	files := []FileEntry{
		{Path: "Q2_K/amd.Instella-MoE-16B-A3B-Think.f16.gguf.Q2_K.gguf", Size: 100},
		{Path: "Q4_K_M/amd.Instella-MoE-16B-A3B-Think.f16.gguf.Q4_K_M.gguf", Size: 200},
		{Path: "Q8_0/amd.Instella-MoE-16B-A3B-Think.f16.gguf.Q8_0.gguf", Size: 300},
	}
	groups, _, _ := GroupFiles(files)
	if len(groups) != 3 {
		t.Fatalf("want 3 groups, got %d: %v", len(groups), labels(groups))
	}
	want := []string{"Q2_K", "Q4_K_M", "Q8_0"}
	for i, q := range want {
		if groups[i].Quant != q || groups[i].Label != q {
			t.Errorf("group %d = quant=%q label=%q, want %q", i, groups[i].Quant, groups[i].Label, q)
		}
		if groups[i].Quant == "F16" || groups[i].Label == "GGUF" {
			t.Errorf("must not collapse requantized files to F16/GGUF: %+v", groups[i])
		}
	}
}

func TestGroupFilesSplitSet(t *testing.T) {
	files := []FileEntry{
		{Path: "model-IQ4_XS-00001-of-00003.gguf", Size: 100},
		{Path: "model-IQ4_XS-00002-of-00003.gguf", Size: 100},
		{Path: "model-IQ4_XS-00003-of-00003.gguf", Size: 100},
		{Path: "README.md", Size: 10},
	}
	groups, _, _ := GroupFiles(files)
	if len(groups) != 1 {
		t.Fatalf("want 1 group, got %d: %+v", len(groups), groups)
	}
	g := groups[0]
	if !g.Split || g.Parts != 3 || len(g.Files) != 3 {
		t.Errorf("split set malformed: %+v", g)
	}
	if g.TotalBytes != 300 {
		t.Errorf("total = %d, want 300", g.TotalBytes)
	}
}

func TestGroupFilesVisionPairing(t *testing.T) {
	files := []FileEntry{
		{Path: "llava-Q4_K_M.gguf", Size: 4000},
		{Path: "mmproj-llava-f16.gguf", Size: 600},
	}
	groups, projectors, _ := GroupFiles(files)
	if len(groups) != 1 {
		t.Fatalf("want exactly one model group (no vision variant), got %+v", groups)
	}
	if groups[0].Vision || len(groups[0].Files) != 1 {
		t.Errorf("base group must not include the projector: %+v", groups[0])
	}
	if len(projectors) != 1 || projectors[0].Path != "mmproj-llava-f16.gguf" {
		t.Errorf("projector must be returned separately: %+v", projectors)
	}
}

func TestGroupFilesProjectorOnlyRepo(t *testing.T) {
	files := []FileEntry{
		{Path: "mmproj-a-f16.gguf", Size: 600},
		{Path: "mmproj-b-f16.gguf", Size: 700},
	}
	groups, projectors, _ := GroupFiles(files)
	if len(groups) != 1 || !groups[0].Vision || len(groups[0].Files) != 2 {
		t.Fatalf("projector-only repo should offer one projector group: %+v", groups)
	}
	if len(projectors) != 0 {
		t.Errorf("projectors already in the group must not be returned again: %+v", projectors)
	}
}

func TestGroupFilesMixedMTPSameQuant(t *testing.T) {
	// DavidAU-style: plain + MTP (+ AMD/LOW MTP) share a quant token.
	files := []FileEntry{
		{Path: "Qwen3.6-NEO-IQ4_XS.gguf", Size: 1000},
		{Path: "Qwen3.6-NEO-MTP-IQ4_XS.gguf", Size: 1100},
		{Path: "Qwen3.6-NEO-AMD-MTP-IQ4_XS.gguf", Size: 1200},
		{Path: "Qwen3.6-NEO-LOW-MTP-IQ4_XS.gguf", Size: 1150},
		{Path: "Qwen3.6-NEO-Q4_K_M.gguf", Size: 2000},
		{Path: "Qwen3.6-NEO-MTP-Q4_K_M.gguf", Size: 2100},
		{Path: "mmproj-F16.gguf", Size: 600},
	}
	groups, projectors, _ := GroupFiles(files)
	if len(projectors) != 1 {
		t.Fatalf("projectors: %+v", projectors)
	}
	if len(groups) != 6 {
		t.Fatalf("want 6 distinct groups (no quant merging across MTP), got %d: %+v", len(groups), labels(groups))
	}
	byQuantMTP := map[string]int{}
	for _, g := range groups {
		byQuantMTP[g.Quant+"|"+g.MTP]++
		if g.Quant == "IQ4_XS" && g.MTP == "" && g.Label != "IQ4_XS" {
			t.Errorf("plain IQ4_XS label = %q", g.Label)
		}
		if g.MTP == "mtp" && !strings.Contains(g.Label, "MTP") {
			t.Errorf("MTP group label missing MTP: %q", g.Label)
		}
		if strings.Contains(g.Files[0].Path, "AMD-MTP") && !strings.Contains(g.Label, "AMD") {
			t.Errorf("AMD MTP label = %q", g.Label)
		}
		if strings.Contains(g.Files[0].Path, "LOW-MTP") && !strings.Contains(g.Label, "LOW") {
			t.Errorf("LOW MTP label = %q", g.Label)
		}
		if len(g.Files) != 1 {
			t.Errorf("group %q should be a single file, got %d", g.Label, len(g.Files))
		}
	}
	if byQuantMTP["IQ4_XS|"] != 1 || byQuantMTP["IQ4_XS|mtp"] != 3 {
		t.Errorf("IQ4_XS split wrong: %v", byQuantMTP)
	}
	if byQuantMTP["Q4_K_M|"] != 1 || byQuantMTP["Q4_K_M|mtp"] != 1 {
		t.Errorf("Q4_K_M split wrong: %v", byQuantMTP)
	}
}

func labels(gs []FileGroup) []string {
	out := make([]string, len(gs))
	for i, g := range gs {
		out[i] = g.Label
	}
	return out
}

func TestGroupFilesExcludesNonGGUF(t *testing.T) {
	files := []FileEntry{
		{Path: "model.safetensors", Size: 1},
		{Path: "config.json", Size: 1},
		{Path: "tokenizer.json", Size: 1},
	}
	if groups, _, _ := GroupFiles(files); len(groups) != 0 {
		t.Fatalf("non-GGUF files must be excluded, got %+v", groups)
	}
}

func TestGroupUniqueIDs(t *testing.T) {
	files := []FileEntry{
		{Path: "a/model-Q4_K_M.gguf", Size: 1},
		{Path: "b/model-Q4_K_M.gguf", Size: 1},
	}
	groups, _, _ := GroupFiles(files)
	if len(groups) != 2 {
		t.Fatalf("want 2 groups, got %d", len(groups))
	}
	if groups[0].ID == groups[1].ID {
		t.Errorf("group IDs must be unique: %q", groups[0].ID)
	}
}

func TestGroupFilesMuseGlimmerCompanions(t *testing.T) {
	files := []FileEntry{
		{Path: "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf", Size: 16756683904},
		{Path: "Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf", Size: 19653960832},
		{Path: "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf", Size: 1400328928},
		{Path: "mmproj-kquant.gguf", Size: 1400328928},
		{Path: "dflash-Muse-Glimmer-30B-Q4_K_M.gguf", Size: 1631208128},
		{Path: "dflash-kquant.gguf", Size: 1631205312},
		{Path: "muse-glimmer-30B-kquant-17gb.gguf", Size: 16756681056},
		{Path: "muse-glimmer-30B-kquant-dynamic.gguf", Size: 19653957984},
	}
	groups, projectors, drafts := GroupFiles(files)
	if len(groups) != 2 {
		t.Fatalf("want 2 trunk groups after alias dedupe, got %d: %v", len(groups), labels(groups))
	}
	for _, g := range groups {
		if g.Draft {
			t.Errorf("trunk group must not be a draft: %+v", g)
		}
		if g.Vision {
			t.Errorf("trunk group must not embed vision: %+v", g)
		}
	}
	if len(projectors) != 1 || !strings.Contains(projectors[0].Path, "mmproj-Muse") {
		t.Fatalf("want canonical mmproj only, got %+v", projectors)
	}
	if len(drafts) != 1 || drafts[0].SpecType != "draft-dflash" {
		t.Fatalf("want one dflash draft, got %+v", drafts)
	}
	if !strings.Contains(drafts[0].Path, "dflash-Muse") {
		t.Errorf("canonical dflash name: %+v", drafts[0])
	}
	quants := map[string]bool{}
	for _, g := range groups {
		quants[g.Quant] = true
	}
	if !quants["Q4_K_M"] || !quants["Q4_K_XL"] {
		t.Errorf("quants = %v", quants)
	}
}

func TestGroupFilesDraftOnlyRepo(t *testing.T) {
	files := []FileEntry{
		{Path: "dflash-Qwen3-4B-b16.gguf", Size: 800},
		{Path: "eagle3.gguf", Size: 400},
	}
	groups, projectors, drafts := GroupFiles(files)
	if len(projectors) != 0 || len(drafts) != 0 {
		t.Fatalf("draft-only repo should offer groups, not companions: proj=%+v drafts=%+v", projectors, drafts)
	}
	if len(groups) != 2 {
		t.Fatalf("want 2 draft groups, got %d: %v", len(groups), labels(groups))
	}
	for _, g := range groups {
		if !g.Draft {
			t.Errorf("expected draft group: %+v", g)
		}
		if g.Vision {
			t.Errorf("draft must not be vision: %+v", g)
		}
	}
}

func TestGroupFilesDoesNotTreatDFlashTrunkAsDraft(t *testing.T) {
	files := []FileEntry{
		{Path: "Muse-Glimmer-30B-DFlash-ROCmFP4.gguf", Size: 8000},
		{Path: "dflash-ROCmFP4-STRIX.gguf", Size: 400},
		{Path: "mmproj-kquant.gguf", Size: 200},
	}
	groups, projectors, drafts := GroupFiles(files)
	if len(groups) != 1 || groups[0].Draft {
		t.Fatalf("trunk must stay a chat group: %+v", groups)
	}
	if len(drafts) != 1 || drafts[0].SpecType != "draft-dflash" {
		t.Fatalf("sidecar draft: %+v", drafts)
	}
	if len(projectors) != 1 {
		t.Fatalf("projectors: %+v", projectors)
	}
}
