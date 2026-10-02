package workflow

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strings"
)

var (
	nodeIDPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)
	// freeEnumPattern bounds runtime-supplied option values (sampler,
	// scheduler names) to what sd.cpp actually uses.
	freeEnumPattern = regexp.MustCompile(`^[a-z0-9_+. -]{0,40}$`)
)

// Validate checks a graph against the registry and the runtime's
// capabilities. It never panics on hostile input and never looks at disk or
// processes: it is safe to call on every edit.
func Validate(g Graph, reg *Registry, caps Caps) []Issue {
	var issues []Issue
	add := func(sev, code, node, port, param, format string, args ...any) {
		issues = append(issues, Issue{Severity: sev, Code: code, Node: node, Port: port, Param: param, Message: fmt.Sprintf(format, args...)})
	}

	if len(g.Nodes) > MaxNodes {
		add(SeverityError, "graph.limit", "", "", "", "graph has %d nodes; the limit is %d", len(g.Nodes), MaxNodes)
		return issues
	}
	if len(g.Edges) > MaxEdges {
		add(SeverityError, "graph.limit", "", "", "", "graph has %d edges; the limit is %d", len(g.Edges), MaxEdges)
		return issues
	}

	nodes := map[string]Node{}
	specs := map[string]NodeSpec{}
	for _, n := range g.Nodes {
		if !nodeIDPattern.MatchString(n.ID) {
			add(SeverityError, "node.id", n.ID, "", "", "node id %q must be 1-64 letters, digits, '_' or '-'", n.ID)
			continue
		}
		if _, dup := nodes[n.ID]; dup {
			add(SeverityError, "node.duplicate_id", n.ID, "", "", "node id %q is used more than once", n.ID)
			continue
		}
		spec, ok := reg.Get(n.Type)
		if !ok {
			add(SeverityError, "node.unknown_type", n.ID, "", "", "unknown node type %q", n.Type)
			continue
		}
		nodes[n.ID] = n
		specs[n.ID] = spec
		validateParams(n, spec, add)
		validateCapabilities(n, spec, caps, add)
	}

	// Edges: endpoints, direction, types, one wire per input.
	wired := map[Endpoint]bool{}
	for _, e := range g.Edges {
		from, fok := nodes[e.From.Node()]
		to, tok := nodes[e.To.Node()]
		if !fok || !tok {
			missing := e.From.Node()
			if fok {
				missing = e.To.Node()
			}
			add(SeverityError, "edge.endpoint", missing, "", "", "wire refers to missing node %q", missing)
			continue
		}
		if from.ID == to.ID {
			add(SeverityError, "edge.self", from.ID, "", "", "a node cannot feed itself")
			continue
		}
		out, ook := findPort(specs[from.ID].Outputs, e.From.Port())
		in, iok := findPort(specs[to.ID].Inputs, e.To.Port())
		if !ook {
			add(SeverityError, "edge.endpoint", from.ID, e.From.Port(), "", "%s has no output %q", specs[from.ID].Title, e.From.Port())
			continue
		}
		if !iok {
			add(SeverityError, "edge.endpoint", to.ID, e.To.Port(), "", "%s has no input %q", specs[to.ID].Title, e.To.Port())
			continue
		}
		if !compatible(out.Type, in.Type) {
			add(SeverityError, "edge.type", to.ID, in.Name, "", "cannot connect %s to %s: %s output does not match %s input", typeList(out.Type), typeList(in.Type), specs[from.ID].Title, specs[to.ID].Title)
			continue
		}
		if wired[e.To] {
			add(SeverityError, "edge.multiple", to.ID, in.Name, "", "input %q of %s has more than one wire", in.Name, specs[to.ID].Title)
			continue
		}
		wired[e.To] = true
	}

	// Required inputs must be wired.
	for _, n := range g.Nodes {
		spec, ok := specs[n.ID]
		if !ok {
			continue
		}
		for _, in := range spec.Inputs {
			if in.Required && !wired[Endpoint{n.ID, in.Name}] {
				add(SeverityError, "input.missing", n.ID, in.Name, "", "%s needs its %q input connected", spec.Title, in.Name)
			}
		}
	}

	issues = append(issues, cycleIssues(g, nodes)...)
	return issues
}

func findPort(ports []Port, name string) (Port, bool) {
	for _, p := range ports {
		if p.Name == name {
			return p, true
		}
	}
	return Port{}, false
}

// compatible reports whether a wire from an output of type out may enter an
// input accepting in. Any shared type is enough.
func compatible(out, in PortTypes) bool {
	for _, o := range out {
		if in.Has(o) {
			return true
		}
	}
	return false
}

func typeList(p PortTypes) string {
	parts := make([]string, len(p))
	for i, t := range p {
		parts[i] = string(t)
	}
	return strings.Join(parts, "|")
}

type addFn func(sev, code, node, port, param, format string, args ...any)

func validateCapabilities(n Node, spec NodeSpec, caps Caps, add addFn) {
	req := append([]string(nil), spec.Requires...)
	if spec.extraRequires != nil {
		req = append(req, spec.extraRequires(n.Params)...)
	}
	if reason := missingCapability(req, caps); reason != "" {
		add(SeverityError, "capability.missing", n.ID, "", "", "%s cannot run: %s", spec.Title, reason)
	}
}

func validateParams(n Node, spec NodeSpec, add addFn) {
	known := map[string]ParamSpec{}
	for _, ps := range spec.Params {
		known[ps.Name] = ps
	}
	for name := range n.Params {
		if _, ok := known[name]; !ok {
			add(SeverityError, "param.unknown", n.ID, "", name, "%s has no parameter %q", spec.Title, name)
		}
	}
	for _, ps := range spec.Params {
		v, set := n.Params[ps.Name]
		if !set || v == nil {
			if ps.Required {
				add(SeverityError, "param.required", n.ID, "", ps.Name, "%s needs %q set", spec.Title, ps.Name)
			}
			continue
		}
		if code, msg := checkParam(ps, v); msg != "" {
			add(SeverityError, code, n.ID, "", ps.Name, "%s: %q %s", spec.Title, ps.Name, msg)
		}
	}
}

// checkParam returns an issue code and a short reason completing the
// sentence "<param> ..." when v is unacceptable for ps, or "", "" when fine.
func checkParam(ps ParamSpec, v any) (code, msg string) {
	const (
		typeCode  = "param.type"
		rangeCode = "param.range"
		enumCode  = "param.enum"
	)
	switch ps.Kind {
	case ParamInt, ParamSeed:
		f, ok := asFloat(v)
		if !ok || f != math.Trunc(f) || math.Abs(f) > 1<<53 {
			return typeCode, "must be a whole number"
		}
		if ps.Kind == ParamSeed {
			if f < -1 {
				return rangeCode, "must be -1 (random) or a non-negative seed"
			}
			return "", ""
		}
		return rangeCode, checkRange(ps, f)
	case ParamFloat:
		f, ok := asFloat(v)
		if !ok || math.IsNaN(f) || math.IsInf(f, 0) {
			return typeCode, "must be a number"
		}
		return rangeCode, checkRange(ps, f)
	case ParamBool:
		if _, ok := v.(bool); !ok {
			return typeCode, "must be true or false"
		}
	case ParamString:
		s, ok := v.(string)
		if !ok {
			return typeCode, "must be text"
		}
		if len(s) > MaxStringLen {
			return rangeCode, fmt.Sprintf("must be at most %d bytes", MaxStringLen)
		}
	case ParamText:
		s, ok := v.(string)
		if !ok {
			return typeCode, "must be text"
		}
		if len(s) > MaxPromptLen {
			return rangeCode, fmt.Sprintf("must be at most %d bytes", MaxPromptLen)
		}
	case ParamPath:
		s, ok := v.(string)
		if !ok || s == "" {
			return typeCode, "must be a file path"
		}
		if len(s) > MaxPathLen || strings.ContainsRune(s, 0) {
			return typeCode, "is not a valid file path"
		}
	case ParamEnum:
		s, ok := v.(string)
		if !ok {
			return typeCode, "must be text"
		}
		if ps.From != "" {
			if !freeEnumPattern.MatchString(s) {
				return enumCode, "contains characters a sampler or scheduler name never has"
			}
			return "", ""
		}
		for _, o := range ps.Options {
			if o == s {
				return "", ""
			}
		}
		return enumCode, fmt.Sprintf("must be one of %s", strings.Join(ps.Options, ", "))
	case ParamModel:
		m, ok := v.(map[string]any)
		if !ok {
			return typeCode, "must be a model reference"
		}
		id, _ := m["library_id"].(string)
		if id == "" || len(id) > 128 {
			return typeCode, "needs a library_id"
		}
		if name, present := m["name"]; present {
			if s, ok := name.(string); !ok || len(s) > 256 {
				return typeCode, "has an invalid name"
			}
		}
	}
	return "", ""
}

func checkRange(ps ParamSpec, f float64) string {
	if ps.Min != nil && ps.Max != nil && (f < *ps.Min || f > *ps.Max) {
		return fmt.Sprintf("must be between %g and %g", *ps.Min, *ps.Max)
	}
	if ps.Min != nil && f < *ps.Min {
		return fmt.Sprintf("must be at least %g", *ps.Min)
	}
	if ps.Max != nil && f > *ps.Max {
		return fmt.Sprintf("must be at most %g", *ps.Max)
	}
	return ""
}

// asFloat reads any JSON-ish number.
func asFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case float32:
		return float64(x), true
	case int:
		return float64(x), true
	case int32:
		return float64(x), true
	case int64:
		return float64(x), true
	case uint:
		return float64(x), true
	case uint32:
		return float64(x), true
	case uint64:
		return float64(x), true
	case json.Number:
		f, err := x.Float64()
		return f, err == nil
	}
	return 0, false
}

// cycleIssues reports the nodes that sit on a cycle. Kahn's algorithm leaves
// every node on or downstream of a loop; pruning nodes with no wire back into
// that remainder leaves only the loop itself, which is what the canvas should
// mark.
func cycleIssues(g Graph, nodes map[string]Node) []Issue {
	indeg := map[string]int{}
	out := map[string][]string{}
	for id := range nodes {
		indeg[id] = 0
	}
	for _, e := range g.Edges {
		a, b := e.From.Node(), e.To.Node()
		if _, ok := nodes[a]; !ok {
			continue
		}
		if _, ok := nodes[b]; !ok {
			continue
		}
		if a == b {
			continue // reported as edge.self
		}
		out[a] = append(out[a], b)
		indeg[b]++
	}
	var queue []string
	for id, d := range indeg {
		if d == 0 {
			queue = append(queue, id)
		}
	}
	remaining := map[string]bool{}
	for id := range nodes {
		remaining[id] = true
	}
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		delete(remaining, id)
		for _, next := range out[id] {
			indeg[next]--
			if indeg[next] == 0 {
				queue = append(queue, next)
			}
		}
	}
	if len(remaining) == 0 {
		return nil
	}
	// Drop nodes that only lead out of the loop (downstream of it).
	for changed := true; changed; {
		changed = false
		for id := range remaining {
			leadsBack := false
			for _, next := range out[id] {
				if remaining[next] {
					leadsBack = true
					break
				}
			}
			if !leadsBack {
				delete(remaining, id)
				changed = true
			}
		}
	}
	var issues []Issue
	for _, n := range g.Nodes { // graph order keeps the output deterministic
		if remaining[n.ID] {
			issues = append(issues, Issue{Severity: SeverityError, Code: "graph.cycle", Node: n.ID, Message: "this node is part of a loop; a graph must flow one way"})
		}
	}
	return issues
}
