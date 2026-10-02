package workflow

import (
	"fmt"

	"github.com/openinfer/openinfer-studio/internal/mediagen"
)

// fakeEnv is an in-memory Env: planning needs no database or process.
type fakeEnv struct {
	models  map[string]ModelInfo
	files   map[string]FileMeta
	running map[string]mediagen.LoadSettings
}

func newFakeEnv() *fakeEnv {
	return &fakeEnv{
		models: map[string]ModelInfo{
			"m1": {ID: "m1", Name: "flux-2-klein-9b", Path: "/models/flux.gguf", Size: 5 << 30, ModTime: 100},
			"m2": {ID: "m2", Name: "wan2.2", Path: "/models/wan.gguf", Size: 11 << 30, ModTime: 200},
		},
		files: map[string]FileMeta{
			"/loras/film-grain-v2.safetensors": {Size: 144 << 20, ModTime: 10},
			"/loras/other/detail.safetensors":  {Size: 90 << 20, ModTime: 11},
			"/vae/flux2-vae.safetensors":       {Size: 300 << 20, ModTime: 12},
			"/img/fisherman.png":               {Size: 2 << 20, ModTime: 13, Width: 640, Height: 800},
			"/img/odd.webp":                    {Size: 1 << 20, ModTime: 14}, // size unreadable
		},
		running: map[string]mediagen.LoadSettings{},
	}
}

func (e *fakeEnv) ResolveModel(id string) (ModelInfo, error) {
	m, ok := e.models[id]
	if !ok {
		return ModelInfo{}, fmt.Errorf("not found")
	}
	return m, nil
}

func (e *fakeEnv) FileMeta(path string) (FileMeta, error) {
	f, ok := e.files[path]
	if !ok {
		return FileMeta{}, fmt.Errorf("no such file")
	}
	return f, nil
}

func (e *fakeEnv) RunningServer(id string) (mediagen.LoadSettings, bool) {
	s, ok := e.running[id]
	return s, ok
}

// allCaps advertises every capability the built-in nodes can require.
var allCaps = Caps{Known: true, Flags: []string{"vae", "lora-model-dir", "llm", "t5xxl", "clip-l", "clip-g"}}

func node(id, typ string, params map[string]any) Node {
	return Node{ID: id, Type: typ, Params: params}
}

func wire(from, fromPort, to, toPort string) Edge {
	return Edge{From: Endpoint{from, fromPort}, To: Endpoint{to, toPort}}
}

func modelRef(id string) map[string]any {
	return map[string]any{"library_id": id, "name": id}
}

// txt2img is the default graph: checkpoint, two prompts, size, Sample, Save.
func txt2img() Graph {
	return Graph{
		Version: 1,
		Nodes: []Node{
			node("n1", "checkpoint.load", map[string]any{"model": modelRef("m1")}),
			node("n2", "prompt", map[string]any{"text": "Studio portrait of an elderly fisherman"}),
			node("n3", "prompt", map[string]any{"text": "blurry, watermark"}),
			node("n4", "latent.empty", map[string]any{"width": 896, "height": 1152, "batch": 1}),
			node("n5", "sample", map[string]any{"seed": 482910337, "steps": 28, "cfg": 1.0, "guidance": 3.5}),
			node("n6", "image.save", nil),
		},
		Edges: []Edge{
			wire("n1", "model", "n5", "model"),
			wire("n1", "clip", "n5", "clip"),
			wire("n1", "vae", "n5", "vae"),
			wire("n2", "cond", "n5", "positive"),
			wire("n3", "cond", "n5", "negative"),
			wire("n4", "size", "n5", "start"),
			wire("n5", "image", "n6", "image"),
		},
	}
}

// hiresFix extends txt2img: Sample -> Resize x1.5 -> Sample (img2img) -> Save.
func hiresFix() Graph {
	g := txt2img()
	g.Nodes = g.Nodes[:5] // drop n6
	g.Nodes = append(g.Nodes,
		node("n6", "image.resize", map[string]any{"scale": 1.5}),
		node("n7", "sample", map[string]any{"seed": 482910337, "steps": 20, "cfg": 1.0, "guidance": 3.5, "strength": 0.35}),
		node("n8", "image.save", nil),
	)
	g.Edges = g.Edges[:6] // keep everything except n5 -> n6
	g.Edges = append(g.Edges,
		wire("n5", "image", "n6", "image"),
		wire("n6", "image", "n7", "start"),
		wire("n1", "model", "n7", "model"),
		wire("n1", "clip", "n7", "clip"),
		wire("n1", "vae", "n7", "vae"),
		wire("n2", "cond", "n7", "positive"),
		wire("n3", "cond", "n7", "negative"),
		wire("n7", "image", "n8", "image"),
	)
	return g
}
