package mediagen

import (
	"encoding/json"
	"reflect"
	"testing"
)

func defaultsJSON(t *testing.T, d WorkflowDefaults) map[string]any {
	t.Helper()
	raw, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err = json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestWorkflowDefaultsNormalizeNativeSamplingAndExplicitZeroes(t *testing.T) {
	doc := decodeCapabilityFixture(t, `{
		"supported_modes":["img_gen","vid_gen"],"current_mode":"img_gen",
		"defaults_by_mode":{
			"img_gen":{"width":1024,"height":768,"batch_count":1,"strength":0,"sample_params":{"sample_steps":28,"sample_method":"euler","scheduler":"discrete","guidance":{"txt_cfg":0,"img_cfg":9,"distilled_guidance":3.5}}},
			"vid_gen":{"width":832,"height":480,"video_frames":33,"fps":16,"sample_params":{"sample_steps":0,"sample_method":"","guidance":{"txt_cfg":1,"distilled_guidance":0}}},
			"invented":{"width":100}
		},"defaults":{"width":64,"sample_params":{"guidance":{"txt_cfg":99}}}
	}`)
	c := normalizeAPICapabilities(doc, "m", "r")
	image := defaultsJSON(t, c.DefaultsByMode["img_gen"])
	want := map[string]any{"width": float64(1024), "height": float64(768), "batch": float64(1), "strength": float64(0), "steps": float64(28), "sampler": "euler", "scheduler": "discrete", "cfg": float64(0), "guidance": 3.5}
	if !reflect.DeepEqual(image, want) {
		t.Fatalf("image defaults=%v want=%v", image, want)
	}
	video := defaultsJSON(t, c.DefaultsByMode["vid_gen"])
	if video["frames"] != float64(33) || video["fps"] != float64(16) || video["guidance"] != float64(0) || video["steps"] != float64(0) || video["sampler"] != "" {
		t.Fatalf("video defaults=%v", video)
	}
	if _, ok := video["batch"]; ok {
		t.Fatal("missing batch default invented")
	}
	if _, ok := video["scheduler"]; ok {
		t.Fatal("missing scheduler default invented")
	}
	if _, ok := c.DefaultsByMode["invented"]; ok {
		t.Fatal("defaults for unsupported mode advertised")
	}
}

func TestWorkflowDefaultsLegacyMirrorOnlyAppliesToCurrentMode(t *testing.T) {
	for _, tc := range []struct {
		name, fixture string
		want          map[string]map[string]any
	}{
		{"legacy", `{"current_mode":"img_gen","defaults":{"width":512,"strength":0}}`, map[string]map[string]any{"img_gen": {"width": float64(512), "strength": float64(0)}}},
		{"modern modes legacy mirror", `{"supported_modes":["img_gen","vid_gen"],"current_mode":"vid_gen","defaults":{"video_frames":17}}`, map[string]map[string]any{"vid_gen": {"frames": float64(17)}}},
		{"no explicit current mode", `{"defaults":{"width":512}}`, map[string]map[string]any{}},
		{"empty modern defeats mirror", `{"supported_modes":[],"current_mode":"img_gen","defaults":{"width":512}}`, map[string]map[string]any{}},
		{"modern defaults defeat mirror", `{"supported_modes":["img_gen"],"current_mode":"img_gen","defaults_by_mode":{},"defaults":{"width":512}}`, map[string]map[string]any{}},
		{"malformed modern defeats mirror", `{"supported_modes":["img_gen"],"current_mode":"img_gen","defaults_by_mode":null,"defaults":{"width":512}}`, map[string]map[string]any{}},
		{"malformed values omitted", `{"current_mode":"img_gen","defaults":{"width":"512","fps":2.5,"strength":false,"sample_params":{"sample_steps":0}}}`, map[string]map[string]any{"img_gen": {"steps": float64(0)}}},
		{"automatic enum sentinels", `{"current_mode":"img_gen","defaults":{"sample_params":{"sample_method":"default","scheduler":"default"}}}`, map[string]map[string]any{"img_gen": {"sampler": "", "scheduler": ""}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := normalizeAPICapabilities(decodeCapabilityFixture(t, tc.fixture), "m", "r")
			got := map[string]map[string]any{}
			for mode, d := range c.DefaultsByMode {
				got[mode] = defaultsJSON(t, d)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("defaults=%v want=%v", got, tc.want)
			}
		})
	}
}
