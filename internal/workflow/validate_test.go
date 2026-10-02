package workflow

import (
	"encoding/json"
	"strings"
	"testing"
)

func validate(g Graph) []Issue { return Validate(g, NewRegistry(), allCaps) }

func TestDefaultGraphsAreValid(t *testing.T) {
	for name, g := range map[string]Graph{"txt2img": txt2img(), "hires fix": hiresFix()} {
		if issues := validate(g); len(issues) != 0 {
			t.Errorf("%s: unexpected issues %+v", name, issues)
		}
	}
}

func TestValidateRejects(t *testing.T) {
	cases := []struct {
		name string
		edit func(g *Graph)
		code string
		node string
	}{
		{"unknown node type", func(g *Graph) { g.Nodes[0].Type = "checkpoint.teleport" }, "node.unknown_type", "n1"},
		{"duplicate node id", func(g *Graph) { g.Nodes[1].ID = "n1" }, "node.duplicate_id", "n1"},
		{"bad node id", func(g *Graph) { g.Nodes[1].ID = "bad id!" }, "node.id", "bad id!"},
		{"unknown param", func(g *Graph) { g.Nodes[4].Params["turbo"] = true }, "param.unknown", "n5"},
		{"steps out of range", func(g *Graph) { g.Nodes[4].Params["steps"] = 9999 }, "param.range", "n5"},
		{"steps not whole", func(g *Graph) { g.Nodes[4].Params["steps"] = 20.5 }, "param.type", "n5"},
		{"seed below -1", func(g *Graph) { g.Nodes[4].Params["seed"] = -5 }, "param.range", "n5"},
		{"enum value", func(g *Graph) { g.Nodes[4].Params["seed_mode"] = "chaotic" }, "param.enum", "n5"},
		{"sampler name charset", func(g *Graph) { g.Nodes[4].Params["sampler"] = "euler; rm -rf" }, "param.enum", "n5"},
		{"text too long", func(g *Graph) { g.Nodes[1].Params["text"] = strings.Repeat("a", MaxPromptLen+1) }, "param.range", "n2"},
		{"model reference shape", func(g *Graph) { g.Nodes[0].Params["model"] = "flux" }, "param.type", "n1"},
		{"required param missing", func(g *Graph) { delete(g.Nodes[0].Params, "model") }, "param.required", "n1"},
		{"wire to missing node", func(g *Graph) { g.Edges[0].From = Endpoint{"ghost", "model"} }, "edge.endpoint", "ghost"},
		{"wire from missing port", func(g *Graph) { g.Edges[0].From = Endpoint{"n1", "teapot"} }, "edge.endpoint", "n1"},
		{"wire into missing port", func(g *Graph) { g.Edges[0].To = Endpoint{"n5", "teapot"} }, "edge.endpoint", "n5"},
		{"type mismatch", func(g *Graph) { g.Edges[0] = wire("n2", "cond", "n5", "model") }, "edge.type", "n5"},
		{"two wires into one input", func(g *Graph) { g.Edges = append(g.Edges, wire("n2", "cond", "n5", "negative")) }, "edge.multiple", "n5"},
		{"self wire", func(g *Graph) { g.Edges = append(g.Edges, wire("n5", "image", "n5", "start")) }, "edge.self", "n5"},
		{"required input unwired", func(g *Graph) { g.Edges = append(g.Edges[:3], g.Edges[4:]...) }, "input.missing", "n5"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			g := txt2img()
			c.edit(&g)
			issues := validate(g)
			for _, i := range issues {
				if i.Code == c.code && i.Node == c.node {
					return
				}
			}
			t.Fatalf("want %s on %q, got %+v", c.code, c.node, issues)
		})
	}
}

func TestCycleIsReportedOnEveryLoopNode(t *testing.T) {
	g := hiresFix()
	// refine feeds back into the first Sample's start: a loop.
	g.Edges = append(g.Edges[:5], g.Edges[6:]...) // drop n4.size -> n5.start (index 5)
	g.Edges = append(g.Edges, wire("n7", "image", "n5", "start"))
	issues := validate(g)
	var loops []string
	for _, i := range issues {
		if i.Code == "graph.cycle" {
			loops = append(loops, i.Node)
		}
	}
	if len(loops) != 3 { // n5, n6, n7
		t.Fatalf("cycle nodes = %v, want n5 n6 n7; issues %+v", loops, issues)
	}
}

func TestCapabilityGating(t *testing.T) {
	g := txt2img()
	g.Nodes = append(g.Nodes, node("n9", "lora.load", map[string]any{"path": "/loras/x.safetensors"}))
	g.Edges[0] = wire("n1", "model", "n9", "model")
	g.Edges = append(g.Edges, wire("n9", "model", "n5", "model"))

	noLora := Caps{Known: true, Flags: []string{"vae"}}
	issues := Validate(g, NewRegistry(), noLora)
	if !hasIssue(issues, "capability.missing") {
		t.Fatalf("issues = %+v, want capability.missing", issues)
	}
	if msg := issues[0].Message; !strings.Contains(msg, "--lora-model-dir") {
		t.Fatalf("message %q should name the missing flag", msg)
	}

	unknown := Validate(g, NewRegistry(), Caps{})
	if !hasIssue(unknown, "capability.missing") || !strings.Contains(unknown[0].Message, "no stable-diffusion.cpp runtime") {
		t.Fatalf("issues = %+v, want a no-runtime reason", unknown)
	}

	if issues := Validate(g, NewRegistry(), allCaps); len(issues) != 0 {
		t.Fatalf("with the capability: %+v", issues)
	}
}

func TestTextEncoderRoleDecidesTheRequiredFlag(t *testing.T) {
	g := txt2img()
	g.Nodes = append(g.Nodes, node("n9", "textencoder.load", map[string]any{"path": "/te/t5.gguf", "role": "t5xxl"}))
	g.Edges[1] = wire("n9", "clip", "n5", "clip")

	llmOnly := Caps{Known: true, Flags: []string{"llm"}}
	if !hasIssue(Validate(g, NewRegistry(), llmOnly), "capability.missing") {
		t.Fatal("a t5xxl encoder needs --t5xxl, not --llm")
	}
	if issues := Validate(g, NewRegistry(), Caps{Known: true, Flags: []string{"t5xxl"}}); len(issues) != 0 {
		t.Fatalf("t5xxl advertised: %+v", issues)
	}
}

func TestValidateSurvivesHostileInput(t *testing.T) {
	g := Graph{Version: 1, Nodes: []Node{
		{ID: "a", Type: "sample", Params: map[string]any{"steps": "many", "cfg": []any{1}, "seed": map[string]any{}}},
		{ID: "b", Type: "", Params: nil},
	}, Edges: []Edge{{From: Endpoint{"", ""}, To: Endpoint{"a", "a"}}}}
	if issues := validate(g); len(issues) == 0 {
		t.Fatal("hostile graph produced no issues")
	}

	big := Graph{Version: 1}
	for i := 0; i <= MaxNodes; i++ {
		big.Nodes = append(big.Nodes, node("n"+strings.Repeat("x", 1), "prompt", nil))
	}
	if issues := validate(big); len(issues) != 1 || issues[0].Code != "graph.limit" {
		t.Fatalf("oversize graph: %+v", issues)
	}
}

func TestParseGraph(t *testing.T) {
	if _, err := ParseGraph(nil); err == nil {
		t.Error("empty input must fail")
	}
	if _, err := ParseGraph([]byte(`{"version": 2, "nodes": [], "edges": []}`)); err == nil || !strings.Contains(err.Error(), "version") {
		t.Errorf("future version: err = %v", err)
	}
	if _, err := ParseGraph([]byte(`{"version": 1, "nodes": [`)); err == nil {
		t.Error("truncated JSON must fail")
	}
	g, err := ParseGraph([]byte(`{"version": 1, "future_field": true}`))
	if err != nil {
		t.Fatalf("unknown fields must be tolerated: %v", err)
	}
	if g.Nodes == nil || g.Edges == nil {
		t.Error("nodes and edges must decode to empty slices, not nil")
	}
	if _, err := ParseGraph(make([]byte, maxGraphBytes+1)); err == nil {
		t.Error("oversize document must fail")
	}
}

func TestPortTypesJSON(t *testing.T) {
	one, _ := json.Marshal(PortTypes{TypeModel})
	many, _ := json.Marshal(PortTypes{TypeSize, TypeImage})
	if string(one) != `"MODEL"` || string(many) != `["SIZE","IMAGE"]` {
		t.Fatalf("marshal = %s, %s", one, many)
	}
	var a, b PortTypes
	if err := json.Unmarshal([]byte(`"VAE"`), &a); err != nil || len(a) != 1 || a[0] != TypeVAE {
		t.Fatalf("string form: %v %v", a, err)
	}
	if err := json.Unmarshal([]byte(`["SIZE","IMAGE"]`), &b); err != nil || len(b) != 2 {
		t.Fatalf("array form: %v %v", b, err)
	}
	if err := json.Unmarshal([]byte(`42`), &a); err == nil {
		t.Fatal("a number is not a port type")
	}
}
