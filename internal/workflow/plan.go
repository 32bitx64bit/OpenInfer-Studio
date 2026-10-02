package workflow

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/mediagen"
)

// Stage kinds.
const (
	StageGenerate = "generate" // one sd-server img_gen / vid_gen request
	StageResize   = "resize"   // CPU image scaling in the backend
	StageSave     = "save"     // copy an output into the media store
)

// Server actions, relative to what is running now.
const (
	ActionReuse   = "reuse"   // a running server already satisfies the graph
	ActionStart   = "start"   // no server is running for the model
	ActionRestart = "restart" // a server is running with different components
)

// Plan is a validated graph compiled into runnable stages.
type Plan struct {
	Stages   []Stage      `json:"stages"`
	Servers  []ServerNeed `json:"servers"`
	Warnings []Issue      `json:"warnings,omitempty"`
}

// ImageRef points at an image a stage consumes: the output of an earlier
// stage, or a local file the user supplied.
type ImageRef struct {
	Stage string `json:"stage,omitempty"`
	Path  string `json:"path,omitempty"`
}

// Stage is one unit of work. Generate stages are the only ones that touch
// the GPU; each absorbs the loaders, prompts and size wired into its Sample.
type Stage struct {
	ID     string `json:"id"`
	NodeID string `json:"node_id"`
	Kind   string `json:"kind"`
	// Key is the node's content hash: same key, same output. Volatile stages
	// (random seed) must never be served from cache.
	Key      string   `json:"key"`
	Volatile bool     `json:"volatile,omitempty"`
	Deps     []string `json:"deps,omitempty"`
	// Absorbs are the nodes fused into this stage (loaders, prompts, size,
	// input images). They have no work of their own; the canvas shows them
	// with the stage that consumes them.
	Absorbs []string `json:"absorbs,omitempty"`

	Generate *GenerateStage `json:"generate,omitempty"`
	Resize   *ResizeStage   `json:"resize,omitempty"`
	Save     *SaveStage     `json:"save,omitempty"`
}

// GenerateStage is one sd-server request plus the server it needs.
type GenerateStage struct {
	Server   string                  `json:"server"` // ServerNeed.Signature
	Params   mediagen.GenerateParams `json:"params"`
	Init     *ImageRef               `json:"init,omitempty"` // image to start from (img2img)
	SeedMode string                  `json:"seed_mode"`
}

// ResizeStage scales an image. Width and Height are the resolved output size.
type ResizeStage struct {
	Src    ImageRef `json:"src"`
	Mode   string   `json:"mode"`
	Scale  float64  `json:"scale,omitempty"`
	Width  int      `json:"width"`
	Height int      `json:"height"`
	Filter string   `json:"filter"`
}

// SaveStage copies a result into the media store under a prefix.
type SaveStage struct {
	Src    ImageRef `json:"src"`
	Kind   string   `json:"kind"` // image|video
	Prefix string   `json:"prefix"`
}

// ServerNeed is one sd-server the plan uses. Servers are identified by their
// load signature: the model plus the launch inputs the graph controls.
type ServerNeed struct {
	Signature string                 `json:"signature"`
	ModelID   string                 `json:"model_id"`
	ModelName string                 `json:"model_name"`
	Overrides mediagen.LoadOverrides `json:"overrides"`
	Action    string                 `json:"action"`
}

// Options narrow what a plan covers.
type Options struct {
	// Only plans just this node and everything upstream of it ("Run to here").
	Only string
}

// Build validates g and compiles it into a Plan. On any error it returns a
// nil plan and every issue found, so the canvas can mark all of them at once.
func Build(g Graph, reg *Registry, env Env, caps Caps, opts Options) (*Plan, []Issue) {
	issues := Validate(g, reg, caps)
	if HasErrors(issues) {
		return nil, issues
	}
	b := newBuilder(g, reg, env)
	b.issues = issues
	plan := b.build(opts)
	if HasErrors(b.issues) {
		return nil, b.issues
	}
	plan.Warnings = b.warnings()
	return plan, b.issues
}

type imgInfo struct {
	ref  ImageRef
	w, h int
}

type nodeKey struct {
	key      string
	volatile bool
}

type builder struct {
	g      Graph
	reg    *Registry
	env    Env
	nodes  map[string]Node
	specs  map[string]NodeSpec
	in     map[string]map[string]Endpoint // node -> input port -> upstream
	issues []Issue
	seen   map[string]bool // dedupes issues reported from more than one pass

	models  map[string]*ModelInfo // by library id; nil = failed lookup
	files   map[string]*FileMeta  // by path; nil = failed lookup
	keys    map[string]nodeKey
	imgs    map[string]imgInfo // IMAGE-producing node -> where its image lives
	servers []ServerNeed
	stages  []Stage
	stageOf map[string]string // node -> stage id (for nodes that become stages)
	nextID  int
}

func newBuilder(g Graph, reg *Registry, env Env) *builder {
	b := &builder{
		g: g, reg: reg, env: env,
		nodes: map[string]Node{}, specs: map[string]NodeSpec{}, seen: map[string]bool{},
		in:     map[string]map[string]Endpoint{},
		models: map[string]*ModelInfo{}, files: map[string]*FileMeta{},
		keys: map[string]nodeKey{}, imgs: map[string]imgInfo{}, stageOf: map[string]string{},
	}
	for _, n := range g.Nodes {
		b.nodes[n.ID] = n
		b.specs[n.ID], _ = reg.Get(n.Type)
	}
	for _, e := range g.Edges {
		if b.in[e.To.Node()] == nil {
			b.in[e.To.Node()] = map[string]Endpoint{}
		}
		b.in[e.To.Node()][e.To.Port()] = e.From
	}
	return b
}

func (b *builder) errorf(code, node, port, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	k := code + "|" + node + "|" + port + "|" + msg
	if b.seen[k] {
		return
	}
	b.seen[k] = true
	b.issues = append(b.issues, Issue{Severity: SeverityError, Code: code, Node: node, Port: port, Message: msg})
}

func (b *builder) warnf(code, node, format string, args ...any) {
	b.issues = append(b.issues, Issue{Severity: SeverityWarning, Code: code, Node: node, Message: fmt.Sprintf(format, args...)})
}

func (b *builder) warnings() []Issue {
	var out []Issue
	for _, i := range b.issues {
		if i.Severity == SeverityWarning {
			out = append(out, i)
		}
	}
	return out
}

func (b *builder) build(opts Options) *Plan {
	active, ok := b.activeSet(opts)
	plan := &Plan{Stages: []Stage{}, Servers: []ServerNeed{}}
	if !ok {
		return plan
	}
	for _, n := range b.topoOrder() {
		if !active[n.ID] {
			continue
		}
		b.keys[n.ID] = b.computeKey(n)
		switch n.Type {
		case "sample":
			b.sample(n, false)
		case "sample.video":
			b.sample(n, true)
		case "image.load":
			b.imageLoad(n)
		case "image.resize":
			b.resize(n)
		case "image.save":
			b.save(n, "image")
		case "video.save":
			b.save(n, "video")
		}
	}
	for i := range b.stages {
		b.stages[i].Absorbs = b.absorbed(b.stages[i].NodeID)
	}
	plan.Stages = b.stages
	plan.Servers = b.servers
	for i := range plan.Servers {
		plan.Servers[i].Action = b.action(plan.Servers[i])
	}
	return plan
}

// absorbed returns the active nodes upstream of a stage's node that are not
// stages themselves, in graph order.
func (b *builder) absorbed(nodeID string) []string {
	seen := map[string]bool{}
	var walk func(id string)
	walk = func(id string) {
		for _, up := range b.in[id] {
			u := up.Node()
			if seen[u] {
				continue
			}
			if _, isStage := b.stageOf[u]; isStage {
				continue
			}
			seen[u] = true
			walk(u)
		}
	}
	walk(nodeID)
	var out []string
	for _, n := range b.g.Nodes {
		if seen[n.ID] {
			out = append(out, n.ID)
		}
	}
	return out
}

// activeSet returns the nodes a run must evaluate: everything upstream of the
// output nodes, or of opts.Only. ok is false when there is nothing to run.
func (b *builder) activeSet(opts Options) (map[string]bool, bool) {
	var targets []string
	if opts.Only != "" {
		if _, found := b.nodes[opts.Only]; !found {
			b.errorf("plan.only_unknown", opts.Only, "", "cannot run to node %q: it is not in the graph", opts.Only)
			return nil, false
		}
		targets = []string{opts.Only}
	} else {
		for _, n := range b.g.Nodes {
			if b.specs[n.ID].Output {
				targets = append(targets, n.ID)
			}
		}
	}
	if len(targets) == 0 {
		b.warnf("plan.no_output", "", "the graph has no Save node, so there is nothing to run")
		return nil, false
	}
	active := map[string]bool{}
	var walk func(id string)
	walk = func(id string) {
		if active[id] {
			return
		}
		active[id] = true
		for _, up := range b.in[id] {
			walk(up.Node())
		}
	}
	for _, t := range targets {
		walk(t)
	}
	return active, true
}

// topoOrder returns nodes upstream-first. Ties keep graph order so plans are
// deterministic. The graph is already known to be acyclic.
func (b *builder) topoOrder() []Node {
	indeg := map[string]int{}
	out := map[string][]string{}
	for _, n := range b.g.Nodes {
		indeg[n.ID] = 0
	}
	for _, e := range b.g.Edges {
		out[e.From.Node()] = append(out[e.From.Node()], e.To.Node())
		indeg[e.To.Node()]++
	}
	var queue []string
	for _, n := range b.g.Nodes {
		if indeg[n.ID] == 0 {
			queue = append(queue, n.ID)
		}
	}
	var order []Node
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		order = append(order, b.nodes[id])
		for _, next := range out[id] {
			indeg[next]--
			if indeg[next] == 0 {
				queue = append(queue, next)
			}
		}
	}
	return order
}

// ---- parameters -----------------------------------------------------------

type params map[string]any

// effective overlays the node's explicit params on the spec defaults.
func (b *builder) effective(n Node) params {
	p := params{}
	for _, ps := range b.specs[n.ID].Params {
		v, ok := n.Params[ps.Name]
		if str, isStr := v.(string); ok && isStr && str == "" && ps.Default == nil {
			ok = false // a cleared optional text field means "unset", same as absent
		}
		if ok && v != nil {
			p[ps.Name] = v
		} else if ps.Default != nil {
			p[ps.Name] = ps.Default
		}
	}
	return p
}

func (p params) str(name string) string { s, _ := p[name].(string); return s }

func (p params) num(name string) float64 { f, _ := asFloat(p[name]); return f }

func (p params) integer(name string) int { return int(p.num(name)) }

// ---- files and models -----------------------------------------------------

func (b *builder) model(node, libraryID string) *ModelInfo {
	if m, seen := b.models[libraryID]; seen {
		return m
	}
	info, err := b.env.ResolveModel(libraryID)
	if err != nil {
		b.errorf("plan.model_missing", node, "", "model %q is not in your library: %v", libraryID, err)
		b.models[libraryID] = nil
		return nil
	}
	b.models[libraryID] = &info
	return &info
}

func (b *builder) file(node, path string) *FileMeta {
	if !filepath.IsAbs(path) {
		b.errorf("plan.file_path", node, "", "%q must be an absolute path", path)
		return nil
	}
	if m, seen := b.files[path]; seen {
		if m == nil {
			b.errorf("plan.file_missing", node, "", "cannot read %q", path)
		}
		return m
	}
	meta, err := b.env.FileMeta(path)
	if err != nil {
		b.errorf("plan.file_missing", node, "", "cannot read %q: %v", path, err)
		b.files[path] = nil
		return nil
	}
	b.files[path] = &meta
	return &meta
}

// ---- cache keys -----------------------------------------------------------

type fileIdentity struct {
	ID      string `json:"id,omitempty"`
	Path    string `json:"path,omitempty"`
	Size    int64  `json:"size"`
	ModTime int64  `json:"mod_time"`
}

// computeKey hashes everything that decides a node's output: its type, its
// effective parameters, the keys of what feeds it, and the identity of any
// file it reads. Titles, positions and groups are deliberately absent.
func (b *builder) computeKey(n Node) nodeKey {
	spec := b.specs[n.ID]
	p := b.effective(n)
	doc := struct {
		V      int               `json:"v"`
		Type   string            `json:"type"`
		Params params            `json:"params"`
		Inputs map[string]string `json:"inputs,omitempty"`
		Files  []fileIdentity    `json:"files,omitempty"`
	}{V: 1, Type: n.Type, Params: p, Inputs: map[string]string{}}

	volatile := false
	for _, in := range spec.Inputs {
		up, wired := b.in[n.ID][in.Name]
		if !wired {
			continue
		}
		k := b.keys[up.Node()]
		doc.Inputs[in.Name] = up.Port() + "@" + k.key
		volatile = volatile || k.volatile
	}
	switch n.Type {
	case "checkpoint.load":
		if m, ok := p["model"].(map[string]any); ok {
			id, _ := m["library_id"].(string)
			if info := b.model(n.ID, id); info != nil {
				doc.Files = append(doc.Files, fileIdentity{ID: info.ID, Size: info.Size, ModTime: info.ModTime})
			}
		}
	case "textencoder.load", "vae.load", "lora.load", "image.load":
		if path := p.str("path"); path != "" {
			if meta := b.file(n.ID, path); meta != nil {
				doc.Files = append(doc.Files, fileIdentity{Path: path, Size: meta.Size, ModTime: meta.ModTime})
			}
		}
	case "sample", "sample.video":
		if p.num("seed") < 0 {
			volatile = true
		}
	}
	raw, _ := json.Marshal(doc) // map keys marshal sorted: deterministic
	sum := sha256.Sum256(raw)
	return nodeKey{key: hex.EncodeToString(sum[:16]), volatile: volatile}
}

// ---- stages ---------------------------------------------------------------

func (b *builder) newStage(n Node, kind string) *Stage {
	b.nextID++
	k := b.keys[n.ID]
	b.stages = append(b.stages, Stage{
		ID: fmt.Sprintf("s%d", b.nextID), NodeID: n.ID, Kind: kind, Key: k.key, Volatile: k.volatile,
	})
	b.stageOf[n.ID] = b.stages[len(b.stages)-1].ID
	return &b.stages[len(b.stages)-1]
}

func (b *builder) upstream(n Node, port string) (Node, string, bool) {
	up, ok := b.in[n.ID][port]
	if !ok {
		return Node{}, "", false
	}
	return b.nodes[up.Node()], up.Port(), true
}

func (b *builder) imageLoad(n Node) {
	path := b.effective(n).str("path")
	meta := b.file(n.ID, path)
	if meta == nil {
		return
	}
	if meta.Width <= 0 || meta.Height <= 0 {
		b.errorf("plan.image_size", n.ID, "", "cannot read the size of %q; use a png or jpeg, or put a Resize node after it", filepath.Base(path))
		return
	}
	b.imgs[n.ID] = imgInfo{ref: ImageRef{Path: path}, w: meta.Width, h: meta.Height}
}

func (b *builder) resize(n Node) {
	up, _, ok := b.upstream(n, "image")
	if !ok {
		return
	}
	src, ok := b.imgs[up.ID]
	if !ok {
		return // upstream already reported its problem
	}
	p := b.effective(n)
	w, h := 0, 0
	if p.str("mode") == "size" {
		w, h = p.integer("width"), p.integer("height")
	} else {
		w = int(math.Round(float64(src.w) * p.num("scale")))
		h = int(math.Round(float64(src.h) * p.num("scale")))
	}
	if w < 16 || h < 16 || w > 4096 || h > 4096 {
		b.errorf("plan.resize_size", n.ID, "", "Resize would produce %d × %d; the limit is 16 to 4096 on each side", w, h)
		return
	}
	b.warnBatch(n.ID, src.ref)
	st := b.newStage(n, StageResize)
	st.Resize = &ResizeStage{Src: src.ref, Mode: p.str("mode"), Scale: p.num("scale"), Width: w, Height: h, Filter: p.str("filter")}
	if src.ref.Stage != "" {
		st.Deps = []string{src.ref.Stage}
	}
	b.imgs[n.ID] = imgInfo{ref: ImageRef{Stage: st.ID}, w: w, h: h}
}

func (b *builder) save(n Node, kind string) {
	port := "image"
	if kind == "video" {
		port = "video"
	}
	up, _, ok := b.upstream(n, port)
	if !ok {
		return
	}
	var ref ImageRef
	if kind == "video" {
		sid, has := b.stageOf[up.ID]
		if !has {
			return
		}
		ref = ImageRef{Stage: sid}
	} else {
		src, has := b.imgs[up.ID]
		if !has {
			return
		}
		ref = src.ref
	}
	st := b.newStage(n, StageSave)
	st.Save = &SaveStage{Src: ref, Kind: kind, Prefix: b.effective(n).str("prefix")}
	if ref.Stage != "" {
		st.Deps = []string{ref.Stage}
	}
}

// sample fuses a Sample node with everything wired into it.
func (b *builder) sample(n Node, video bool) {
	p := b.effective(n)
	var ov mediagen.LoadOverrides

	// MODEL: walk back through LoRA nodes to the checkpoint.
	var loras []Node
	cur, _, _ := b.upstream(n, "model")
	for cur.Type == "lora.load" {
		loras = append(loras, cur)
		cur, _, _ = b.upstream(cur, "model")
	}
	if cur.Type != "checkpoint.load" {
		b.errorf("plan.model_chain", n.ID, "model", "the model input must come from Load Checkpoint, directly or through Load LoRA")
		return
	}
	checkpoint := cur
	cp := b.effective(checkpoint)
	ref, _ := cp["model"].(map[string]any)
	libraryID, _ := ref["library_id"].(string)
	info := b.model(checkpoint.ID, libraryID)

	// CLIP and VAE: from the same checkpoint (auto-paired companions) or from
	// an explicit loader that overrides the launch input.
	if up, _, ok := b.upstream(n, "clip"); ok {
		switch up.Type {
		case "checkpoint.load":
			if up.ID != checkpoint.ID {
				b.errorf("plan.clip_source", n.ID, "clip", "the text encoder comes from a different checkpoint than the model; use Load Text Encoder to mix them")
			}
		case "textencoder.load":
			ep := b.effective(up)
			path := ep.str("path")
			b.file(up.ID, path)
			switch ep.str("role") {
			case "llm":
				ov.LLM = path
			case "t5xxl":
				ov.T5XXL = path
			case "clip_l":
				ov.ClipL = path
			case "clip_g":
				ov.ClipG = path
			}
		}
	}
	if up, _, ok := b.upstream(n, "vae"); ok {
		switch up.Type {
		case "checkpoint.load":
			if up.ID != checkpoint.ID {
				b.errorf("plan.vae_source", n.ID, "vae", "the VAE comes from a different checkpoint than the model; use Load VAE to mix them")
			}
		case "vae.load":
			path := b.effective(up).str("path")
			b.file(up.ID, path)
			ov.VAE = path
		}
	}

	// Prompts and LoRA tags. sd-server applies LoRAs through <lora:name:w>
	// tags in the prompt, read from one directory fixed at launch.
	positive := ""
	if up, _, ok := b.upstream(n, "positive"); ok {
		positive = b.effective(up).str("text")
	}
	if strings.TrimSpace(positive) == "" {
		b.errorf("plan.prompt_empty", n.ID, "positive", "the positive prompt is empty")
	}
	negative := ""
	if up, _, ok := b.upstream(n, "negative"); ok {
		negative = b.effective(up).str("text")
	}
	var tags []string
	for i := len(loras) - 1; i >= 0; i-- { // checkpoint-side first
		lp := b.effective(loras[i])
		path := lp.str("path")
		b.file(loras[i].ID, path)
		dir := filepath.Dir(path)
		name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		if strings.ContainsAny(name, "<>: \t") {
			b.errorf("plan.lora_name", loras[i].ID, "", "LoRA file name %q cannot be used in a prompt tag; rename it without spaces, ':' or angle brackets", name)
			continue
		}
		if ov.LoraModelDir != "" && ov.LoraModelDir != dir {
			b.errorf("plan.lora_dir", loras[i].ID, "", "all LoRAs on one Sample must be in the same folder (%s and %s differ)", ov.LoraModelDir, dir)
			continue
		}
		ov.LoraModelDir = dir
		tags = append(tags, fmt.Sprintf("<lora:%s:%g>", name, lp.num("strength")))
	}
	prompt := positive
	if len(tags) > 0 {
		prompt = strings.TrimRight(positive, " ") + " " + strings.Join(tags, " ")
	}

	gp := mediagen.GenerateParams{
		Kind:           mediagen.KindImage,
		Prompt:         prompt,
		NegativePrompt: negative,
		Steps:          p.integer("steps"),
		CFGScale:       p.num("cfg"),
		Seed:           int64(p.num("seed")),
		Sampler:        p.str("sampler"),
		Scheduler:      p.str("scheduler"),
		Guidance:       p.num("guidance"),
		OutputFormat:   p.str("output_format"),
	}
	if video {
		gp.Kind = mediagen.KindVideo
		gp.VideoFrames = p.integer("frames")
		gp.FPS = p.integer("fps")
	}

	// start: a size recipe (text-to-image/video) or an image (image-to-image).
	var init *ImageRef
	var deps []string
	if up, _, ok := b.upstream(n, "start"); ok {
		switch up.Type {
		case "latent.empty":
			lp := b.effective(up)
			gp.Width, gp.Height = lp.integer("width"), lp.integer("height")
			if !video {
				gp.BatchCount = lp.integer("batch")
			}
		default: // an IMAGE producer
			if video {
				b.errorf("plan.i2v_unsupported", n.ID, "start", "image-to-video is not available yet; start Sample (video) from an Empty Latent")
				break
			}
			src, found := b.imgs[up.ID]
			if !found {
				break
			}
			gp.Width, gp.Height = src.w, src.h
			gp.BatchCount = 1
			gp.Strength = p.num("strength")
			init = &src.ref
			if src.ref.Path != "" {
				gp.InitImagePath = src.ref.Path
			}
			if src.ref.Stage != "" {
				deps = append(deps, src.ref.Stage)
				b.warnBatch(n.ID, src.ref)
			}
		}
	}

	if info == nil {
		return // model_missing already reported
	}
	sig := signature(info.ID, ov)
	b.useServer(ServerNeed{Signature: sig, ModelID: info.ID, ModelName: info.Name, Overrides: ov})

	st := b.newStage(n, StageGenerate)
	st.Deps = deps
	st.Generate = &GenerateStage{Server: sig, Params: gp, Init: init, SeedMode: p.str("seed_mode")}
	if !video {
		b.imgs[n.ID] = imgInfo{ref: ImageRef{Stage: st.ID}, w: gp.Width, h: gp.Height}
	}
}

// warnBatch notes that a multi-image Sample feeds an image-to-image or resize
// step: only the first image continues down the graph.
func (b *builder) warnBatch(node string, src ImageRef) {
	prev := b.stageByID(src.Stage)
	if prev != nil && prev.Generate != nil && prev.Generate.Params.BatchCount > 1 {
		b.warnf("plan.batch_downstream", node, "the upstream Sample makes %d images; only the first continues to this node", prev.Generate.Params.BatchCount)
	}
}

func (b *builder) stageByID(id string) *Stage {
	for i := range b.stages {
		if b.stages[i].ID == id {
			return &b.stages[i]
		}
	}
	return nil
}

// signature identifies a server by its model and the launch inputs the graph
// controls. Everything else about the launch is the user's own configuration.
func signature(modelID string, ov mediagen.LoadOverrides) string {
	raw, _ := json.Marshal(struct {
		Model string                 `json:"model"`
		Ov    mediagen.LoadOverrides `json:"ov"`
	}{modelID, ov})
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:8])
}

func (b *builder) useServer(s ServerNeed) {
	for _, have := range b.servers {
		if have.Signature == s.Signature {
			return
		}
	}
	b.servers = append(b.servers, s)
}

// action compares a need with what is running now.
func (b *builder) action(s ServerNeed) string {
	running, ok := b.env.RunningServer(s.ModelID)
	if !ok {
		return ActionStart
	}
	if s.Overrides.SatisfiedBy(running) {
		return ActionReuse
	}
	return ActionRestart
}
