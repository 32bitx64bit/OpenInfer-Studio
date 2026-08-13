package huggingface

import "testing"

func TestGroupFilesSortedByQuant(t *testing.T) {
	files := []FileEntry{
		{Path: "m-Q8_0.gguf", Size: 8000},
		{Path: "m-Q4_K_M.gguf", Size: 4000},
		{Path: "m-F16.gguf", Size: 16000},
		{Path: "m-Q2_K.gguf", Size: 2000},
		{Path: "m-Q6_K.gguf", Size: 6000},
	}
	groups, _, _ := GroupFiles(files)
	want := []string{"Q2_K", "Q4_K_M", "Q6_K", "Q8_0", "F16"}
	if len(groups) != len(want) {
		t.Fatalf("got %d groups: %+v", len(groups), groups)
	}
	for i, q := range want {
		if groups[i].Quant != q {
			t.Errorf("position %d = %s, want %s (all: %v)", i, groups[i].Quant, q,
				func() []string {
					var qs []string
					for _, g := range groups {
						qs = append(qs, g.Quant)
					}
					return qs
				}())
		}
	}
}

func TestProjectorsNotDuplicated(t *testing.T) {
	files := []FileEntry{
		{Path: "m-Q8_0.gguf", Size: 8000},
		{Path: "m-Q4_K_M.gguf", Size: 4000},
		{Path: "mmproj-m-f16.gguf", Size: 600},
	}
	groups, projectors, _ := GroupFiles(files)
	if len(groups) != 2 {
		t.Fatalf("want 2 groups (one per quant), got %+v", groups)
	}
	for _, g := range groups {
		if g.Vision {
			t.Errorf("no vision variant groups expected: %+v", g)
		}
	}
	if groups[0].Quant != "Q4_K_M" || groups[1].Quant != "Q8_0" {
		t.Errorf("quant order wrong: %+v", groups)
	}
	if len(projectors) != 1 {
		t.Errorf("projectors returned once: %+v", projectors)
	}
}
