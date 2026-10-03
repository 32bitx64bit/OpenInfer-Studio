package workflow

import (
	"encoding/json"
	"testing"
)

func TestRegistryIsConsistent(t *testing.T) {
	reg := NewRegistry()
	seen := map[string]bool{}
	for _, s := range reg.Specs() {
		if seen[s.Type] {
			t.Errorf("duplicate node type %s", s.Type)
		}
		seen[s.Type] = true
		if s.Title == "" || s.Category == "" {
			t.Errorf("%s: missing title or category", s.Type)
		}
		ports := map[string]bool{}
		for _, p := range append(append([]Port{}, s.Inputs...), s.Outputs...) {
			if len(p.Type) == 0 {
				t.Errorf("%s.%s: no port type", s.Type, p.Name)
			}
			for _, pt := range p.Type {
				known := false
				for _, a := range AllPortTypes {
					known = known || a == pt
				}
				if !known {
					t.Errorf("%s.%s: unknown port type %s", s.Type, p.Name, pt)
				}
			}
			_ = ports
		}
		for _, ps := range s.Params {
			if ps.Default == nil {
				continue
			}
			if code, msg := checkParam(ps, ps.Default); msg != "" {
				t.Errorf("%s.%s: default %v fails its own check (%s: %s)", s.Type, ps.Name, ps.Default, code, msg)
			}
		}
	}
}

func TestViewMarksUnavailableNodes(t *testing.T) {
	reg := NewRegistry()
	byType := func(views []NodeTypeView, typ string) NodeTypeView {
		for _, v := range views {
			if v.Type == typ {
				return v
			}
		}
		t.Fatalf("no view for %s", typ)
		return NodeTypeView{}
	}

	none := reg.View(Caps{})
	if v := byType(none, "lora.load"); v.Available || v.Reason == "" {
		t.Fatalf("lora.load without a runtime: %+v", v)
	}
	if v := byType(none, "sample"); !v.Available {
		t.Fatal("nodes with no requirements are always available")
	}

	partial := reg.View(Caps{Known: true, Flags: []string{"vae"}})
	if v := byType(partial, "vae.load"); !v.Available {
		t.Fatal("vae.load should be available when --vae is advertised")
	}
	if v := byType(partial, "lora.load"); v.Available {
		t.Fatal("lora.load should be unavailable without advertised structured API support")
	}
}

func TestNodeTypeViewJSONMatchesTheDocumentedShape(t *testing.T) {
	var sample NodeTypeView
	for _, v := range NewRegistry().View(allCaps) {
		if v.Type == "sample" {
			sample = v
		}
	}
	raw, err := json.Marshal(sample)
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	_ = json.Unmarshal(raw, &doc)
	for _, key := range []string{"type", "category", "title", "inputs", "outputs", "params", "available"} {
		if _, ok := doc[key]; !ok {
			t.Errorf("descriptor missing %q: %s", key, raw)
		}
	}
	inputs := doc["inputs"].([]any)
	start := inputs[len(inputs)-1].(map[string]any)
	if start["name"] != "start" {
		t.Fatalf("last input = %v", start)
	}
	if types, ok := start["type"].([]any); !ok || len(types) != 2 {
		t.Fatalf("start must accept SIZE and IMAGE as an array, got %v", start["type"])
	}
	// private hook must never leak into the contract
	if _, leaked := doc["extraRequires"]; leaked {
		t.Fatal("extraRequires leaked into JSON")
	}
}

func TestDescriptorListsAreNeverNull(t *testing.T) {
	for _, v := range NewRegistry().View(allCaps) {
		raw, _ := json.Marshal(v)
		var doc map[string]json.RawMessage
		_ = json.Unmarshal(raw, &doc)
		for _, key := range []string{"inputs", "outputs", "params"} {
			if string(doc[key]) == "null" {
				t.Errorf("%s.%s marshals as null; QML iterates it", v.Type, key)
			}
		}
	}
}
