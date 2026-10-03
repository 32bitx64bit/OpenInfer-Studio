package workflow

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/mediagen"
)

func mustPlan(t *testing.T, g Graph, env *fakeEnv, opts Options) *Plan {
	t.Helper()
	plan, issues := Build(g, NewRegistry(), env, allCaps, opts)
	if plan == nil {
		t.Fatalf("Build returned no plan; issues: %+v", issues)
	}
	return plan
}

func hasIssue(issues []Issue, code string) bool {
	for _, i := range issues {
		if i.Code == code {
			return true
		}
	}
	return false
}

// formParams reproduces what the current Image Studio form sends and how
// POST /models/{id}/media/generate turns it into GenerateParams (it merges the
// body over {Seed: -1}; see internal/api/media.go generateMedia).
func formParams(t *testing.T, body string) mediagen.GenerateParams {
	t.Helper()
	merged, _ := json.Marshal(mediagen.GenerateParams{Seed: -1})
	base := map[string]json.RawMessage{}
	if err := json.Unmarshal(merged, &base); err != nil {
		t.Fatal(err)
	}
	raw := map[string]json.RawMessage{}
	if err := json.Unmarshal([]byte(body), &raw); err != nil {
		t.Fatal(err)
	}
	for k, v := range raw {
		base[k] = v
	}
	final, _ := json.Marshal(base)
	var p mediagen.GenerateParams
	if err := json.Unmarshal(final, &p); err != nil {
		t.Fatal(err)
	}
	return p
}

// The default graph must ask sd-server for exactly what the Simple form does.
func TestDefaultTxt2ImgMatchesTheFormRequest(t *testing.T) {
	plan := mustPlan(t, txt2img(), newFakeEnv(), Options{})

	if len(plan.Stages) != 2 || plan.Stages[0].Kind != StageGenerate || plan.Stages[1].Kind != StageSave {
		t.Fatalf("stages = %+v, want generate then save", plan.Stages)
	}
	want := formParams(t, `{
		"kind": "image", "prompt": "Studio portrait of an elderly fisherman",
		"negative_prompt": "blurry, watermark", "width": 896, "height": 1152,
		"steps": 28, "cfg_scale": 1, "seed": 482910337, "sampler": "", "scheduler": "",
		"guidance": 3.5, "output_format": "png", "batch_count": 1
	}`)
	got := plan.Stages[0].Generate.Params
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("graph request differs from form request\n got:  %+v\n want: %+v", got, want)
	}
	// And the sd-server body built from either is identical.
	a, b := mediagen.SDRequestBody(got), mediagen.SDRequestBody(want)
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if string(ja) != string(jb) {
		t.Fatalf("sd-server bodies differ:\n%s\n%s", ja, jb)
	}
}

func TestImg2ImgFromFileMatchesTheFormRequest(t *testing.T) {
	g := Graph{Version: 1,
		Nodes: []Node{
			node("n1", "checkpoint.load", map[string]any{"model": modelRef("m1")}),
			node("n2", "prompt", map[string]any{"text": "oil painting"}),
			node("n3", "image.load", map[string]any{"path": "/img/fisherman.png"}),
			node("n5", "sample", map[string]any{"seed": 5, "steps": 20, "cfg": 7.0, "strength": 0.6}),
			node("n6", "image.save", nil),
		},
		Edges: []Edge{
			wire("n1", "model", "n5", "model"), wire("n1", "clip", "n5", "clip"),
			wire("n2", "cond", "n5", "positive"), wire("n3", "image", "n5", "start"),
			wire("n5", "image", "n6", "image"),
		},
	}
	plan := mustPlan(t, g, newFakeEnv(), Options{})
	want := formParams(t, `{
		"kind": "image", "prompt": "oil painting", "negative_prompt": "",
		"width": 640, "height": 800, "steps": 20, "cfg_scale": 7, "seed": 5,
		"sampler": "", "scheduler": "", "guidance": 0, "output_format": "png",
		"batch_count": 1, "strength": 0.6, "init_image_path": "/img/fisherman.png"
	}`)
	if got := plan.Stages[0].Generate.Params; !reflect.DeepEqual(got, want) {
		t.Fatalf("img2img request differs\n got:  %+v\n want: %+v", got, want)
	}
}

func TestHiresFixFusesIntoThreeStagesOnOneServer(t *testing.T) {
	plan := mustPlan(t, hiresFix(), newFakeEnv(), Options{})

	kinds := []string{}
	for _, s := range plan.Stages {
		kinds = append(kinds, s.Kind)
	}
	if strings.Join(kinds, ",") != "generate,resize,generate,save" {
		t.Fatalf("stage kinds = %v", kinds)
	}
	if len(plan.Servers) != 1 {
		t.Fatalf("servers = %+v, want exactly one (same load signature)", plan.Servers)
	}
	resize := plan.Stages[1]
	if resize.Resize.Width != 1344 || resize.Resize.Height != 1728 {
		t.Fatalf("resize = %+v, want 1344 x 1728", resize.Resize)
	}
	refine := plan.Stages[2].Generate
	if refine.Init == nil || refine.Init.Stage != resize.ID {
		t.Fatalf("refine init = %+v, want stage %s", refine.Init, resize.ID)
	}
	if refine.Params.Width != 1344 || refine.Params.Height != 1728 || refine.Params.Strength != 0.35 {
		t.Fatalf("refine params = %+v", refine.Params)
	}
	if plan.Stages[2].Generate.Server != plan.Stages[0].Generate.Server {
		t.Fatal("both Samples must share one server signature")
	}
	if len(plan.Stages[2].Deps) != 1 || plan.Stages[2].Deps[0] != resize.ID {
		t.Fatalf("refine deps = %v, want [%s]", plan.Stages[2].Deps, resize.ID)
	}
}

func TestRunToHerePlansOnlyUpstream(t *testing.T) {
	plan := mustPlan(t, hiresFix(), newFakeEnv(), Options{Only: "n6"})
	if len(plan.Stages) != 2 || plan.Stages[1].Kind != StageResize {
		t.Fatalf("stages = %+v, want generate + resize only", plan.Stages)
	}
}

func TestRunToUnknownNodeIsAnError(t *testing.T) {
	plan, issues := Build(txt2img(), NewRegistry(), newFakeEnv(), allCaps, Options{Only: "nope"})
	if plan != nil || !hasIssue(issues, "plan.only_unknown") {
		t.Fatalf("plan=%v issues=%+v, want plan.only_unknown", plan, issues)
	}
}

func TestGraphWithoutSaveHasNothingToRun(t *testing.T) {
	g := txt2img()
	g.Nodes = g.Nodes[:5]
	g.Edges = g.Edges[:6]
	plan, issues := Build(g, NewRegistry(), newFakeEnv(), allCaps, Options{})
	if plan == nil || len(plan.Stages) != 0 {
		t.Fatalf("plan = %+v, want an empty plan", plan)
	}
	if !hasIssue(issues, "plan.no_output") {
		t.Fatalf("issues = %+v, want plan.no_output warning", issues)
	}
}

func TestLoraBecomesStructuredNativeInput(t *testing.T) {
	g := txt2img()
	g.Nodes = append(g.Nodes, node("n9", "lora.load", map[string]any{"path": "/loras/film-grain-v2.safetensors", "strength": 0.65}))
	g.Edges[0] = wire("n1", "model", "n9", "model")
	g.Edges = append(g.Edges, wire("n9", "model", "n5", "model"))

	plan := mustPlan(t, g, newFakeEnv(), Options{})
	gen := plan.Stages[0].Generate
	if gen.Params.Prompt != "Studio portrait of an elderly fisherman" {
		t.Fatalf("prompt = %q", gen.Params.Prompt)
	}
	if len(gen.Params.Lora) != 1 || gen.Params.Lora[0].Path != "/loras/film-grain-v2.safetensors" || gen.Params.Lora[0].Multiplier != 0.65 {
		t.Fatalf("structured LoRA: %+v", gen.Params.Lora)
	}
	if plan.Servers[0].Overrides.LoraModelDir != "/loras" {
		t.Fatal("selected files must be registered in the native LoRA catalog")
	}
}

func TestStructuredLorasMayLiveInSharedCatalogSubfolders(t *testing.T) {
	g := txt2img()
	g.Nodes = append(g.Nodes,
		node("n9", "lora.load", map[string]any{"path": "/loras/film-grain-v2.safetensors"}),
		node("n10", "lora.load", map[string]any{"path": "/loras/other/detail.safetensors"}))
	g.Edges[0] = wire("n1", "model", "n9", "model")
	g.Edges = append(g.Edges, wire("n9", "model", "n10", "model"), wire("n10", "model", "n5", "model"))

	plan, issues := Build(g, NewRegistry(), newFakeEnv(), allCaps, Options{})
	if plan == nil || HasErrors(issues) || len(plan.Stages[0].Generate.Params.Lora) != 2 {
		t.Fatalf("structured native LoRAs: plan=%v issues=%+v", plan, issues)
	}
	if plan.Servers[0].Overrides.LoraModelDir != "/loras" {
		t.Fatal("nested LoRAs must share their common catalog tree")
	}
}

func TestExplicitVAEAndEncoderBecomeOverrides(t *testing.T) {
	env := newFakeEnv()
	env.files["/te/qwen3.gguf"] = FileMeta{Size: 5 << 30, ModTime: 20}
	g := txt2img()
	g.Nodes = append(g.Nodes,
		node("n9", "vae.load", map[string]any{"path": "/vae/flux2-vae.safetensors"}),
		node("n10", "textencoder.load", map[string]any{"path": "/te/qwen3.gguf", "role": "llm"}))
	g.Edges[1] = wire("n10", "clip", "n5", "clip")
	g.Edges[2] = wire("n9", "vae", "n5", "vae")

	ov := mustPlan(t, g, env, Options{}).Servers[0].Overrides
	if ov.VAE != "/vae/flux2-vae.safetensors" || ov.LLM != "/te/qwen3.gguf" {
		t.Fatalf("overrides = %+v", ov)
	}
}

func TestClipFromAnotherCheckpointIsRejected(t *testing.T) {
	g := txt2img()
	g.Nodes = append(g.Nodes, node("n9", "checkpoint.load", map[string]any{"model": modelRef("m2")}))
	g.Edges[1] = wire("n9", "clip", "n5", "clip")
	plan, issues := Build(g, NewRegistry(), newFakeEnv(), allCaps, Options{})
	if plan != nil || !hasIssue(issues, "plan.clip_source") {
		t.Fatalf("plan=%v issues=%+v, want plan.clip_source", plan, issues)
	}
}

func TestPlanProblemsAreAllReportedTogether(t *testing.T) {
	g := txt2img()
	g.Nodes[0].Params["model"] = modelRef("ghost")
	g.Nodes[1].Params["text"] = "  "
	plan, issues := Build(g, NewRegistry(), newFakeEnv(), allCaps, Options{})
	if plan != nil {
		t.Fatal("plan must be nil when problems exist")
	}
	if !hasIssue(issues, "plan.model_missing") || !hasIssue(issues, "plan.prompt_empty") {
		t.Fatalf("issues = %+v, want both model_missing and prompt_empty", issues)
	}
}

func TestUnreadableImageSizeIsReported(t *testing.T) {
	g := Graph{Version: 1,
		Nodes: []Node{
			node("n1", "checkpoint.load", map[string]any{"model": modelRef("m1")}),
			node("n2", "prompt", map[string]any{"text": "x"}),
			node("n3", "image.load", map[string]any{"path": "/img/odd.webp"}),
			node("n5", "sample", nil),
			node("n6", "image.save", nil),
		},
		Edges: []Edge{
			wire("n1", "model", "n5", "model"), wire("n1", "clip", "n5", "clip"),
			wire("n2", "cond", "n5", "positive"), wire("n3", "image", "n5", "start"),
			wire("n5", "image", "n6", "image"),
		},
	}
	plan, issues := Build(g, NewRegistry(), newFakeEnv(), allCaps, Options{})
	if plan != nil || !hasIssue(issues, "plan.image_size") {
		t.Fatalf("plan=%v issues=%+v, want plan.image_size", plan, issues)
	}
}

func TestImageToVideoRequiresAdvertisedSupport(t *testing.T) {
	g := Graph{Version: 1,
		Nodes: []Node{
			node("n1", "checkpoint.load", map[string]any{"model": modelRef("m2")}),
			node("n2", "prompt", map[string]any{"text": "x"}),
			node("n3", "image.load", map[string]any{"path": "/img/fisherman.png"}),
			node("n5", "sample.video", nil),
			node("n6", "video.save", nil),
		},
		Edges: []Edge{
			wire("n1", "model", "n5", "model"), wire("n1", "clip", "n5", "clip"),
			wire("n2", "cond", "n5", "positive"), wire("n3", "image", "n5", "start"),
			wire("n5", "video", "n6", "video"),
		},
	}
	plan, issues := Build(g, NewRegistry(), newFakeEnv(), allCaps, Options{})
	if plan != nil || !hasIssue(issues, "plan.api_capability") {
		t.Fatalf("plan=%v issues=%+v, want plan.i2v_unsupported", plan, issues)
	}
}

func TestTextToVideoPlan(t *testing.T) {
	g := Graph{Version: 1,
		Nodes: []Node{
			node("n1", "checkpoint.load", map[string]any{"model": modelRef("m2")}),
			node("n2", "prompt", map[string]any{"text": "waves"}),
			node("n4", "latent.empty", map[string]any{"width": 832, "height": 480}),
			node("n5", "sample.video", map[string]any{"frames": 33, "fps": 16}),
			node("n6", "video.save", nil),
		},
		Edges: []Edge{
			wire("n1", "model", "n5", "model"), wire("n1", "clip", "n5", "clip"),
			wire("n2", "cond", "n5", "positive"), wire("n4", "size", "n5", "start"),
			wire("n5", "video", "n6", "video"),
		},
	}
	plan := mustPlan(t, g, newFakeEnv(), Options{})
	p := plan.Stages[0].Generate.Params
	if p.Kind != mediagen.KindVideo || p.VideoFrames != 33 || p.FPS != 16 || p.OutputFormat != "webm" || p.BatchCount != 0 {
		t.Fatalf("video params = %+v", p)
	}
	if plan.Stages[1].Save.Kind != "video" || plan.Stages[1].Save.Src.Stage != plan.Stages[0].ID {
		t.Fatalf("save stage = %+v", plan.Stages[1])
	}
}

func TestServerActions(t *testing.T) {
	env := newFakeEnv()
	if a := mustPlan(t, txt2img(), env, Options{}).Servers[0].Action; a != ActionStart {
		t.Fatalf("nothing running: action = %q, want start", a)
	}

	env.running["m1"] = mediagen.LoadSettings{FlashAttention: true}
	if a := mustPlan(t, txt2img(), env, Options{}).Servers[0].Action; a != ActionReuse {
		t.Fatalf("running, graph overrides nothing: action = %q, want reuse", a)
	}

	g := txt2img()
	g.Nodes = append(g.Nodes, node("n9", "vae.load", map[string]any{"path": "/vae/flux2-vae.safetensors"}))
	g.Edges[2] = wire("n9", "vae", "n5", "vae")
	if a := mustPlan(t, g, env, Options{}).Servers[0].Action; a != ActionRestart {
		t.Fatalf("running with a different VAE: action = %q, want restart", a)
	}

	env.running["m1"] = mediagen.LoadSettings{VAE: "/vae/flux2-vae.safetensors"}
	if a := mustPlan(t, g, env, Options{}).Servers[0].Action; a != ActionReuse {
		t.Fatalf("running with the same VAE: action = %q, want reuse", a)
	}
}

func TestSignatureDependsOnModelAndOverridesOnly(t *testing.T) {
	a := mustPlan(t, txt2img(), newFakeEnv(), Options{}).Servers[0].Signature

	g := txt2img()
	g.Nodes[4].Params["steps"] = 50
	g.Nodes[1].Params["text"] = "a different prompt"
	if b := mustPlan(t, g, newFakeEnv(), Options{}).Servers[0].Signature; b != a {
		t.Fatal("sampler settings and prompts must not change the load signature")
	}

	g = txt2img()
	g.Nodes[0].Params["model"] = modelRef("m2")
	if b := mustPlan(t, g, newFakeEnv(), Options{}).Servers[0].Signature; b == a {
		t.Fatal("a different model must change the load signature")
	}
}

func TestCacheKeys(t *testing.T) {
	key := func(g Graph, env *fakeEnv) string {
		plan := mustPlan(t, g, env, Options{})
		return plan.Stages[0].Key
	}
	base := txt2img()
	base.Nodes[4].Params["seed"] = 11 // a fixed seed is cacheable

	k0 := key(base, newFakeEnv())

	moved := txt2img()
	moved.Nodes[4].Params["seed"] = 11
	moved.Nodes[4].Pos = [2]float64{999, 999}
	moved.Nodes[1].Title = "renamed"
	moved.Groups = []Group{{Title: "g", Nodes: []string{"n5"}}}
	if key(moved, newFakeEnv()) != k0 {
		t.Fatal("position, title and groups must not change a cache key")
	}

	edited := txt2img()
	edited.Nodes[4].Params["seed"] = 11
	edited.Nodes[1].Params["text"] = "a different prompt"
	if key(edited, newFakeEnv()) == k0 {
		t.Fatal("an upstream prompt edit must change the downstream key")
	}

	explicitDefault := txt2img() // QML sends "" after the user clears a dropdown
	explicitDefault.Nodes[4].Params["seed"] = 11
	explicitDefault.Nodes[4].Params["scheduler"] = ""
	if key(explicitDefault, newFakeEnv()) != k0 {
		t.Fatal("a cleared optional field must hash the same as an unset one")
	}

	env := newFakeEnv()
	m := env.models["m1"]
	m.ModTime++
	env.models["m1"] = m
	if key(base, env) == k0 {
		t.Fatal("a model file that changed on disk must change the key")
	}
}

func TestRandomSeedIsVolatileAndPropagatesDownstream(t *testing.T) {
	g := hiresFix()
	g.Nodes[4].Params["seed"] = -1 // base Sample random; refine keeps a fixed seed
	plan := mustPlan(t, g, newFakeEnv(), Options{})
	for _, s := range plan.Stages {
		if !s.Volatile {
			t.Fatalf("stage %s (%s) must be volatile when an upstream seed is random", s.ID, s.Kind)
		}
	}

	fixed := mustPlan(t, hiresFix(), newFakeEnv(), Options{})
	for _, s := range fixed.Stages {
		if s.Volatile {
			t.Fatalf("stage %s must be cacheable when every seed is fixed", s.ID)
		}
	}
}

func TestBatchDownstreamWarns(t *testing.T) {
	g := hiresFix()
	g.Nodes[3].Params["batch"] = 4
	plan, issues := Build(g, NewRegistry(), newFakeEnv(), allCaps, Options{})
	if plan == nil || !hasIssue(issues, "plan.batch_downstream") {
		t.Fatalf("plan=%v issues=%+v, want a batch_downstream warning", plan, issues)
	}
	if len(plan.Warnings) != 1 {
		t.Fatalf("warnings = %+v", plan.Warnings)
	}
}
