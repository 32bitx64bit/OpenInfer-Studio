package orchestrate

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"quantlab/core"
)

// Capabilities is the probed feature set of one tool binary, derived from its
// own --help output. Planners must only emit flags the binary advertises.
type Capabilities struct {
	Tool Tool   `json:"tool"`
	Path string `json:"path"`
	// Version is the binary-reported version string, when discoverable.
	Version string `json:"version,omitempty"`
	// Flags holds every advertised long or short option ("--imatrix", "-t").
	Flags map[string]bool `json:"flags"`
	// Types lists quant type tokens advertised in the help text (best
	// effort; llama-quantize enumerates them in its usage output).
	Types []string `json:"types,omitempty"`
}

// Has reports whether the binary advertises flag.
func (c Capabilities) Has(flag string) bool { return c.Flags[flag] }

// HasType reports whether the binary advertises quant type token t (e.g.
// "Q4_K") in its help output. An empty Types list means "unknown" (older
// tools); callers must treat that as "no type gating possible", not as
// "nothing supported".
func (c Capabilities) HasType(t string) bool {
	for _, x := range c.Types {
		if x == t {
			return true
		}
	}
	return false
}

var (
	flagTokenRe  = regexp.MustCompile(`(?m)(?:^|[\s\[|,])(-{1,2}[a-zA-Z][a-zA-Z0-9_-]*)`)
	versionToken = regexp.MustCompile(`(?i)version[:=\s]+([0-9A-Za-z][0-9A-Za-z.+-]*)`)
)

// ParseHelp extracts advertised flags and a version string from a tool's
// --help text. It accepts output from either stream.
func ParseHelp(tool Tool, path, help string) Capabilities {
	c := Capabilities{Tool: tool, Path: path, Flags: map[string]bool{}}
	for _, m := range flagTokenRe.FindAllStringSubmatch(help, -1) {
		c.Flags[m[1]] = true
	}
	if m := versionToken.FindStringSubmatch(help); m != nil {
		c.Version = m[1]
	}
	seen := map[string]bool{}
	for _, tok := range strings.FieldsFunc(help, func(r rune) bool {
		return r == ' ' || r == '\t' || r == '\n' || r == ',' || r == '|' || r == '[' || r == ']'
	}) {
		d := core.DType(tok)
		if d.Valid() && d.IsQuant() && !seen[tok] {
			seen[tok] = true
			c.Types = append(c.Types, tok)
		}
	}
	sort.Strings(c.Types)
	return c
}

// ProbeCapabilities runs `<path> --help` through r and parses the advertised
// flags. Help output may arrive on stdout or stderr with any exit code, so
// both streams are combined and exit status is not enforced.
func ProbeCapabilities(ctx context.Context, r Runner, tool Tool, path string) (Capabilities, error) {
	iv := Invocation{Tool: tool, Path: path, Argv: []string{"--help"}, Env: []string{}}
	res, err := r.Run(ctx, iv)
	if err != nil {
		var ee *ExitError
		if !errors.As(err, &ee) {
			return Capabilities{}, fmt.Errorf("orchestrate: probe %s: %w", tool, err)
		}
	}
	text := CombinedOutput(res)
	if strings.TrimSpace(text) == "" {
		return Capabilities{}, fmt.Errorf("orchestrate: probe %s: empty --help output", tool)
	}
	caps := ParseHelp(tool, path, text)
	// --version is authoritative when the help text carried no version.
	if caps.Version == "" && caps.Has("--version") {
		vres, verr := r.Run(ctx, Invocation{Tool: tool, Path: path, Argv: []string{"--version"}, Env: []string{}})
		if verr == nil {
			if m := versionToken.FindStringSubmatch(CombinedOutput(vres)); m != nil {
				caps.Version = m[1]
			}
		}
	}
	return caps, nil
}
