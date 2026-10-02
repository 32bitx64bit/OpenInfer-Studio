package pipeline

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"quantlab/anchor"
	"quantlab/core"
	"quantlab/profile"
	"quantlab/tensorbank"
)

// Data-free token-embedding floor. The policy floor on an untied embedding
// table (Q6_K at comfortable budgets) exists for rare-token rows a short
// corpus never exercises. Every row's quantization error is computable
// directly, so the floor is relaxed to the cheapest rung whose worst rows
// are no noisier than a typical row at the compression target. It never
// rises above the policy floor, and tied tables (the output head) are left
// alone.

const (
	embedFloorVersion = 1
	// embedRowSampleElements caps the elements scored per rung; larger
	// tables are row-strided (uniform over the vocabulary, so the p99.9
	// still rests on tens of thousands of rows).
	embedRowSampleElements = 64 << 20
)

type embedFloorRecord struct {
	Tensor string     `json:"tensor"`
	Policy core.DType `json:"policyFloor"`
	Floor  core.DType `json:"floor"`
	// Yardstick is the median row error interpolated at the target bpw.
	Yardstick float64                              `json:"yardstick"`
	Stats     map[core.DType]profile.RowErrorStats `json:"stats"`
}

type embedFloorState struct {
	Version   int                         `json:"version"`
	Signature string                      `json:"signature"`
	Records   map[string]embedFloorRecord `json:"records"`
}

func (e *Engine) embedFloorPath() string { return filepath.Join(e.workDir(), "embed-floor.json") }

// embedRowFloorEnabled reports whether the per-row embedding check runs.
func (e *Engine) embedRowFloorEnabled() bool {
	if e.Extra.NoEmbedRowFloor || e.DryRun {
		return false
	}
	return e.effortProfile().EmbedRowFloor && e.effectiveTargetBPW() > 0
}

// deriveAnchors is the run's anchor set: the bpw policy, with the untied
// token-embedding floor relaxed by the per-row check when enabled.
func (e *Engine) deriveAnchors(bank *core.TensorBank) (*anchor.Set, error) {
	set, err := anchor.Derive(bank, nil, anchor.PolicyForBPW(e.Run.Config.TargetBPW))
	if err != nil {
		return nil, err
	}
	if e.embedRowFloorEnabled() {
		if err := e.applyEmbedRowFloors(bank, set); err != nil {
			// Fail-open: the policy floor is the conservative answer.
			e.printf("  embed floor: kept policy floor (%v)\n", err)
		}
	}
	return set, nil
}

func (e *Engine) embedFloorSignature(bank *core.TensorBank, scored []core.DType) (string, error) {
	sha, err := e.payloadIdentitySHA()
	if err != nil {
		return "", err
	}
	h := sha256.New()
	fmt.Fprintf(h, "v%d\x00%s\x00bpw=%g\x00cap=%d\x00", embedFloorVersion, sha, e.effectiveTargetBPW(), embedRowSampleElements)
	for _, d := range scored {
		fmt.Fprintf(h, "%s\x00", d)
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

// embedRowCandidates are the scored rungs the run may also assign.
func (e *Engine) embedRowCandidates() []core.DType {
	allowed := map[core.DType]bool{}
	for _, d := range e.candidateDTypes() {
		allowed[d.BaseTensorType()] = true
	}
	var out []core.DType
	for _, d := range profile.EmbedRowDTypes {
		if allowed[d] {
			out = append(out, d)
		}
	}
	return out
}

func (e *Engine) applyEmbedRowFloors(bank *core.TensorBank, set *anchor.Set) error {
	var targets []core.Anchor
	for _, a := range set.Hard {
		if a.Kind == core.AnchorExplicit && a.Reason == anchor.ReasonTokenEmbeddings && a.Name != "" {
			targets = append(targets, a)
		}
	}
	if len(targets) == 0 {
		return nil
	}
	scored := e.embedRowCandidates()
	if len(scored) == 0 {
		return nil
	}
	sig, err := e.embedFloorSignature(bank, scored)
	if err != nil {
		return err
	}
	st := e.loadEmbedFloorState(sig)
	dirty := false
	for _, a := range targets {
		rec, ok := st.Records[a.Name]
		if !ok || rec.Policy != a.MinDType {
			rec, err = e.scoreEmbedRows(a, scored)
			if err != nil {
				return err
			}
			st.Records[a.Name] = rec
			dirty = true
			e.printEmbedFloor(rec)
		}
		if rec.Floor != "" && rec.Floor != rec.Policy {
			set.RelaxEmbeddingFloor(a.Name, rec.Floor, "per-row check")
		}
	}
	if dirty {
		return e.writeJSON(e.embedFloorPath(), st)
	}
	return nil
}

func (e *Engine) loadEmbedFloorState(sig string) *embedFloorState {
	fresh := &embedFloorState{Version: embedFloorVersion, Signature: sig, Records: map[string]embedFloorRecord{}}
	data, err := os.ReadFile(e.embedFloorPath())
	if err != nil {
		return fresh
	}
	var st embedFloorState
	if json.Unmarshal(data, &st) != nil || st.Version != embedFloorVersion || st.Signature != sig || st.Records == nil {
		return fresh
	}
	return &st
}

// scoreEmbedRows reads the embedding table from the payload source and
// picks its relaxed floor.
func (e *Engine) scoreEmbedRows(a core.Anchor, scored []core.DType) (embedFloorRecord, error) {
	rec := embedFloorRecord{Tensor: a.Name, Policy: a.MinDType, Floor: a.MinDType}
	vals, ne0, err := readEmbedRows(e.payloadSource(), a.Name, embedRowSampleElements)
	if err != nil {
		return rec, err
	}
	if vals == nil {
		return rec, nil // not a float table: nothing to score
	}
	rec.Stats = profile.EmbeddingRowErrors(vals, ne0, scored)
	y, ok := profile.YardstickError(e.effectiveTargetBPW(), rec.Stats)
	if !ok {
		return rec, nil
	}
	rec.Yardstick = y
	if d, ok := profile.ChooseEmbeddingFloor(rec.Stats, y, a.MinDType, anchor.Rank); ok {
		rec.Floor = d
	}
	return rec, nil
}

func (e *Engine) printEmbedFloor(rec embedFloorRecord) {
	fs := rec.Stats[rec.Floor]
	if rec.Floor == rec.Policy {
		e.printf("  embed floor: %s keeps %s (no cheaper rung's p99.9 row error <= target-rate median %.4f)\n",
			rec.Tensor, rec.Policy, rec.Yardstick)
		return
	}
	e.printf("  embed floor: %s %s -> %s (p99.9 row error %.4f <= target-rate median %.4f, %d rows)\n",
		rec.Tensor, rec.Policy, rec.Floor, fs.P999, rec.Yardstick, fs.Rows)
}

// readEmbedRows decodes a float 2-D tensor's rows, row-strided down to at
// most maxElems elements. Returns nil values for non-float tensors.
func readEmbedRows(path, name string, maxElems uint64) ([]float32, int, error) {
	src, err := tensorbank.OpenSource(path)
	if err != nil {
		return nil, 0, err
	}
	defer src.Close()
	file, err := tensorbank.Parse(src)
	if err != nil {
		return nil, 0, err
	}
	ti, ok := file.FindTensor(name)
	if !ok {
		return nil, 0, fmt.Errorf("embedding %s not in payload", name)
	}
	var esz uint64
	switch ti.DType {
	case core.DTypeF32:
		esz = 4
	case core.DTypeF16, core.DTypeBF16:
		esz = 2
	default:
		return nil, 0, nil
	}
	if len(ti.Shape) != 2 || ti.Shape[0] == 0 || ti.Shape[1] == 0 {
		return nil, 0, nil
	}
	ne0, rows := ti.Shape[0], ti.Shape[1]
	stride := uint64(1)
	if ne0*rows > maxElems && maxElems > 0 {
		stride = (ne0*rows + maxElems - 1) / maxElems
	}
	keep := (rows + stride - 1) / stride
	vals := make([]float32, keep*ne0)
	buf := make([]byte, ne0*esz)
	base := file.PayloadOffset(ti)
	for i := uint64(0); i < keep; i++ {
		r := i * stride
		if _, err := src.ReadAt(buf, base+int64(r*ne0*esz)); err != nil {
			return nil, 0, fmt.Errorf("read %s row %d: %w", name, r, err)
		}
		profile.DecodeFloats(vals[i*ne0:(i+1)*ne0], buf, ti.DType)
	}
	return vals, int(ne0), nil
}
