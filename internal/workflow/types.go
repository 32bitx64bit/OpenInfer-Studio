// Package workflow is the node-graph engine behind Image Studio's Graph view.
//
// A graph is plain JSON that QML edits. This package owns everything that
// decides anything about it: the registry of node types, validation, and the
// planner that fuses a graph into the coarse requests sd-server can actually
// run (sd-server samples and decodes in one call, so there are no latent
// wires and each Sample node absorbs the loaders, prompts and size wired
// into it). Nothing here talks to a process; the executor drives mediagen.
package workflow

import (
	"encoding/json"
	"fmt"
)

// PortType is the kind of value a wire carries. Wires only join matching
// types; the same rule is enforced live in QML and again by Validate.
type PortType string

const (
	TypeModel  PortType = "MODEL"  // diffusion checkpoint plus its LoRA stack
	TypeClip   PortType = "CLIP"   // text encoder(s)
	TypeVAE    PortType = "VAE"    // decoder weights
	TypeCond   PortType = "COND"   // a prompt; text, not a tensor
	TypeSize   PortType = "SIZE"   // width, height, batch: a recipe, not a latent
	TypeImage  PortType = "IMAGE"  // a stored image file
	TypeVideo  PortType = "VIDEO"  // a stored video file
	TypeMask   PortType = "MASK"   // inpainting region (reserved)
	TypeScalar PortType = "SCALAR" // int, float, string, seed (reserved)
)

// AllPortTypes lists every port type in palette order.
var AllPortTypes = []PortType{TypeModel, TypeClip, TypeVAE, TypeCond, TypeSize, TypeImage, TypeVideo, TypeMask, TypeScalar}

// PortTypes is the set of types a port accepts or emits. It marshals as a
// bare string for one type and as an array for several, so descriptors read
// "type": "MODEL" or "type": ["SIZE", "IMAGE"].
type PortTypes []PortType

func (p PortTypes) MarshalJSON() ([]byte, error) {
	if len(p) == 1 {
		return json.Marshal(string(p[0]))
	}
	return json.Marshal([]PortType(p))
}

func (p *PortTypes) UnmarshalJSON(b []byte) error {
	var one PortType
	if err := json.Unmarshal(b, &one); err == nil {
		*p = PortTypes{one}
		return nil
	}
	var many []PortType
	if err := json.Unmarshal(b, &many); err != nil {
		return fmt.Errorf("port type must be a string or an array of strings")
	}
	*p = PortTypes(many)
	return nil
}

// Has reports whether t is one of the types.
func (p PortTypes) Has(t PortType) bool {
	for _, x := range p {
		if x == t {
			return true
		}
	}
	return false
}

// Port is one socket on a node.
type Port struct {
	Name     string    `json:"name"`
	Type     PortTypes `json:"type"`
	Required bool      `json:"required,omitempty"`
}

// ParamKind selects the widget QML draws and the checks Validate applies.
type ParamKind string

const (
	ParamInt    ParamKind = "int"
	ParamFloat  ParamKind = "float"
	ParamString ParamKind = "string"
	ParamText   ParamKind = "text" // multi-line
	ParamBool   ParamKind = "bool"
	ParamEnum   ParamKind = "enum"
	ParamSeed   ParamKind = "seed"  // integer >= -1; -1 is random
	ParamModel  ParamKind = "model" // {"library_id": "...", "name": "..."}
	ParamPath   ParamKind = "path"  // absolute local file path
)

// ParamSpec describes one editable parameter of a node type.
type ParamSpec struct {
	Name     string    `json:"name"`
	Kind     ParamKind `json:"kind"`
	Label    string    `json:"label,omitempty"`
	Required bool      `json:"required,omitempty"`
	Default  any       `json:"default,omitempty"`
	Min      *float64  `json:"min,omitempty"`
	Max      *float64  `json:"max,omitempty"`
	Options  []string  `json:"options,omitempty"`
	// From names a runtime-supplied option list (e.g. "capabilities.samplers").
	// Enum values are then free-form and the UI fills the dropdown.
	From string `json:"from,omitempty"`
	// ShowWhen is a UI hint ("start is IMAGE"); the validator ignores it.
	ShowWhen string `json:"show_when,omitempty"`
}

// NodeSpec is the declarative description of a node type. QML draws sockets,
// widgets, palette and inspector from it, so a new node type needs no QML.
type NodeSpec struct {
	APIFeature  string      `json:"api_feature,omitempty"`
	Type        string      `json:"type"`
	Category    string      `json:"category"`
	Title       string      `json:"title"`
	Description string      `json:"description,omitempty"`
	Inputs      []Port      `json:"inputs"`
	Outputs     []Port      `json:"outputs"`
	Params      []ParamSpec `json:"params"`
	// Requires lists runtime capability ids (sd-server --help) the node needs.
	Requires []string `json:"requires,omitempty"`
	// Output marks nodes a run is anchored on (Save Image, Save Video).
	Output bool `json:"output,omitempty"`

	// extraRequires adds capabilities that depend on the node's parameters.
	extraRequires func(params map[string]any) []string
}

// Graph is the stored and exported workflow document.
type Graph struct {
	Version int      `json:"version"`
	Name    string   `json:"name,omitempty"`
	Nodes   []Node   `json:"nodes"`
	Edges   []Edge   `json:"edges"`
	Groups  []Group  `json:"groups,omitempty"`
	View    *ViewBox `json:"view,omitempty"`
}

// Node is one placed node. Params holds only values the user set; the
// registry supplies defaults for the rest.
type Node struct {
	Collapsed bool           `json:"collapsed,omitempty"`
	ID        string         `json:"id"`
	Type      string         `json:"type"`
	Title     string         `json:"title,omitempty"`
	Pos       [2]float64     `json:"pos"`
	Params    map[string]any `json:"params,omitempty"`
}

// Endpoint addresses one socket: [nodeID, portName].
type Endpoint [2]string

// Node returns the node id.
func (e Endpoint) Node() string { return e[0] }

// Port returns the port name.
func (e Endpoint) Port() string { return e[1] }

// Edge is a wire from an output socket to an input socket.
type Edge struct {
	From Endpoint `json:"from"`
	To   Endpoint `json:"to"`
}

// Group is a labelled frame around nodes. UI only; never part of a cache key.
type Group struct {
	Color string   `json:"color,omitempty"`
	Note  string   `json:"note,omitempty"`
	Title string   `json:"title"`
	Nodes []string `json:"nodes"`
}

// ViewBox is the saved pan and zoom. UI only.
type ViewBox struct {
	X    float64 `json:"x"`
	Y    float64 `json:"y"`
	Zoom float64 `json:"zoom"`
}

// Severity of an Issue.
const (
	SeverityError   = "error"
	SeverityWarning = "warning"
)

// Issue is one problem found in a graph, addressed to a node (and port or
// parameter) so the canvas can mark the exact socket.
type Issue struct {
	Severity string `json:"severity"`
	Code     string `json:"code"`
	Node     string `json:"node,omitempty"`
	Port     string `json:"port,omitempty"`
	Param    string `json:"param,omitempty"`
	Message  string `json:"message"`
}

// HasErrors reports whether any issue is an error.
func HasErrors(issues []Issue) bool {
	for _, i := range issues {
		if i.Severity == SeverityError {
			return true
		}
	}
	return false
}

// Limits keep a pathological or hostile document from reaching the planner.
const (
	MaxNodes      = 500
	MaxEdges      = 2000
	MaxPromptLen  = 8000
	MaxStringLen  = 1024
	MaxPathLen    = 4096
	GraphVersion  = 1
	maxGraphBytes = 2 << 20
)

// ParseGraph decodes and structurally checks a stored graph document. It is
// lenient about unknown fields (a newer app may have saved them) and strict
// about version and size. Semantic checks belong to Validate.
func ParseGraph(raw []byte) (Graph, error) {
	if len(raw) == 0 {
		return Graph{}, fmt.Errorf("graph is empty")
	}
	if len(raw) > maxGraphBytes {
		return Graph{}, fmt.Errorf("graph exceeds %d MiB", maxGraphBytes>>20)
	}
	var g Graph
	if err := json.Unmarshal(raw, &g); err != nil {
		return Graph{}, fmt.Errorf("invalid graph JSON: %w", err)
	}
	if g.Version != GraphVersion {
		return Graph{}, fmt.Errorf("unsupported graph version %d (this build reads version %d)", g.Version, GraphVersion)
	}
	if len(g.Nodes) > MaxNodes {
		return Graph{}, fmt.Errorf("graph has %d nodes; the limit is %d", len(g.Nodes), MaxNodes)
	}
	if len(g.Edges) > MaxEdges {
		return Graph{}, fmt.Errorf("graph has %d edges; the limit is %d", len(g.Edges), MaxEdges)
	}
	if g.Nodes == nil {
		g.Nodes = []Node{}
	}
	if g.Edges == nil {
		g.Edges = []Edge{}
	}
	return g, nil
}
