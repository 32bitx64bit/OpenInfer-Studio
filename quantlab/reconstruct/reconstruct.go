// Package reconstruct rewrites a float GGUF into an equivalent float GGUF
// that quantizes more cleanly at a ~3.5 bpw budget:
//
//   - IGH: absorb RMSNorm γ into the following projections, then apply a
//     residual-wide normalized FWHT-256. The rotated model is algebraically
//     equivalent under standard llama.cpp RMSNorm + RoPE + matmul, so the
//     output stays a loadable GGUF.
//   - CSK: ridge least-squares fit of each SwiGLU down-projection against a
//     Q3_K probe of (gate, up), compensating next-layer error in-weight.
//
// Assemble order is permute (original residual basis) then Hadamard
// (FWHT-256). MagR pins super-weights in F32; there is no unloadable FP16
// sidecar. FreqVQ and expert centroid are skip-safe extras, not the quality
// claim for a 3.5 bpw hybrid.
//
// Both stages keep GGUF layout and dtype. When outPath is the source path,
// payloads are rewritten in place (no second model-sized file).
package reconstruct

import (
	"context"
	"fmt"
	"path/filepath"

	"quantlab/profile"
	"quantlab/tensorbank"
)

// Options selects reconstruct stages. All stages keep GGUF layout and dtype,
// so they can rewrite a job-private source in place (outPath == src) without
// a second model-sized file.
//
// Assemble order is permute → Hadamard → MagR → LWC → CSK → FreqVQ →
// expert centroid. Permute runs in the original residual basis; Hadamard
// (FWHT-256) follows so the two are not silently composed in the wrong order.
type Options struct {
	Permute        bool
	Hadamard       bool
	MagR           bool
	LWC            bool
	CSK            bool
	FreqVQ         bool
	ExpertCentroid bool
	Imatrix        map[string]profile.ImatrixStats
	Context        context.Context
	// MaxWorkingSetBytes bounds CSK's estimated in-memory layer workspace.
	// Zero uses a conservative 8 GiB ceiling.
	MaxWorkingSetBytes uint64
	Progress           func(Progress)
}

func (o Options) any() bool {
	return o.Permute || o.Hadamard || o.MagR || o.LWC || o.CSK || o.FreqVQ || o.ExpertCentroid
}

// Progress reports bounded reconstruction work without exposing tensor data.
type Progress struct {
	Phase   string
	Layer   int
	Layers  int
	Current int
	Total   int
	Detail  string
}

// Result reports what was written.
type Result struct {
	Written      bool
	Hadamard     bool
	CSKLayers    int
	ResidualDim  int
	SkipHadamard string
	SkipCSK      string
	InPlace      bool
}

func samePath(a, b string) bool {
	if a == "" || b == "" {
		return a == b
	}
	aa, err1 := filepath.Abs(a)
	bb, err2 := filepath.Abs(b)
	return err1 == nil && err2 == nil && aa == bb
}

func rewriteGGUF(ctx context.Context, src *tensorbank.Source, outPath string, fn tensorbank.PayloadFunc) error {
	if src != nil && (outPath == "" || samePath(src.Path(), outPath)) {
		return tensorbank.RewriteInPlace(ctx, src, fn)
	}
	return tensorbank.RewriteContext(ctx, src, outPath, fn)
}

// Apply writes outPath as a GGUF-native reconstructed model. When outPath is
// empty or equal to src.Path(), payloads are rewritten in place and the file
// size does not change. When neither stage applies, Written is false and
// outPath is left untouched.
func Apply(src *tensorbank.Source, outPath string, opts Options) (Result, error) {
	var res Result
	if src == nil {
		return res, fmt.Errorf("reconstruct: nil source")
	}
	if !opts.any() {
		return res, nil
	}
	ctx := opts.Context
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return res, err
	}
	inPlace := outPath == "" || samePath(src.Path(), outPath)
	if inPlace {
		outPath = src.Path()
		res.InPlace = true
	}
	current := src
	var extra *tensorbank.Source
	defer func() {
		if extra != nil {
			extra.Close()
		}
	}()
	advance := func(applied bool) error {
		if !applied {
			return nil
		}
		res.Written = true
		if samePath(current.Path(), outPath) {
			return nil
		}
		s, err := tensorbank.OpenSource(outPath)
		if err != nil {
			return err
		}
		if extra != nil {
			extra.Close()
		}
		extra = s
		current = s
		return nil
	}

	if opts.Permute {
		pr, err := applyPermute(ctx, current, outPath, opts.Imatrix)
		if err != nil {
			return res, err
		}
		if err := advance(pr.applied); err != nil {
			return res, err
		}
	}
	if opts.Hadamard {
		hr, err := applyHadamard(ctx, current, outPath)
		if err != nil {
			return res, err
		}
		res.Hadamard = hr.applied
		res.ResidualDim = hr.dim
		res.SkipHadamard = hr.reason
		if err := advance(hr.applied); err != nil {
			return res, err
		}
	}
	if opts.MagR {
		mr, err := applyMagR(ctx, current, outPath, opts.Imatrix)
		if err != nil {
			return res, err
		}
		if err := advance(mr.applied); err != nil {
			return res, err
		}
	}
	if opts.LWC {
		lr, err := applyLWC(ctx, current, outPath, opts.Imatrix)
		if err != nil {
			return res, err
		}
		if err := advance(lr.applied); err != nil {
			return res, err
		}
	}
	if opts.CSK {
		cr, err := applyCSK(ctx, current, outPath, opts)
		if err != nil {
			return res, err
		}
		res.CSKLayers = cr.layers
		res.SkipCSK = cr.reason
		if err := advance(cr.applied); err != nil {
			return res, err
		}
	}
	if opts.FreqVQ {
		fr, err := applyFreqVQ(ctx, current, outPath, opts.Imatrix)
		if err != nil {
			return res, err
		}
		if err := advance(fr.applied); err != nil {
			return res, err
		}
	}
	if opts.ExpertCentroid {
		er, err := applyCentroid(ctx, current, outPath, opts.Imatrix)
		if err != nil {
			return res, err
		}
		if err := advance(er.applied); err != nil {
			return res, err
		}
	}
	return res, nil
}
