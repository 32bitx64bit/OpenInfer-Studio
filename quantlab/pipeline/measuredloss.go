package pipeline

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"quantlab/core"
	"quantlab/orchestrate"
	"quantlab/profile"
	"quantlab/qtype"
	"quantlab/tensorbank"
)

// Measured loss table: per-tensor, per-dtype importance-weighted SSE
// measured with the run's own llama-quantize on a row-sampled copy of the
// payload. The quantity matches profile.BuildExactLossTableCfg exactly
// (Σ_rows Σ_c imp_c (w_rc − Q_d(w)_rc)² extrapolated from sampled rows),
// so everything downstream — Sensitivity.Loss, the uncalibrated estimator,
// persistence — is unchanged.
//
// Measuring with the real tool is ~10× faster than the Go exact table on
// large models and, critically, prices IQ rungs by what llama-quantize
// actually writes instead of the approximate Go proxies.

const measuredSampleElements = 256 << 20

// goFallbackDTypes are dtypes whose Go reference codec is exact after the
// Phase 0 fixes. When measurement fails for one of these, the Go table is
// a faithful fallback. All other dtypes must never be priced with the Go
// proxies (IQ2/IQ3/IQ1 proxies are 1.3–5× pessimistic and lock those
// rungs out of the solver).
var goFallbackDTypes = map[core.DType]bool{
	core.DTypeQ2_K:   true,
	core.DTypeQ3_K:   true,
	core.DTypeQ4_K_T: true,
	core.DTypeQ5_K_T: true,
	core.DTypeQ6_K:   true,
	core.DTypeIQ4_NL: true,
	core.DTypeIQ4_XS: true,
	core.DTypeQ8_0:   true,
}

// measuredTensor is one eligible tensor's sampling geometry.
type measuredTensor struct {
	desc core.TensorDesc
	keep []uint64 // row indices per slice (same for every expert slice)
	S    uint64   // expert slices (Π Shape[2:]), 1 for 2-D
	R    uint64   // original row count (Shape[1])
}

// buildMeasuredLossTable measures per-tensor, per-dtype wSSE using the
// run's llama-quantize on a row-sampled payload. Returns the table, the
// dtypes that could not be measured (and must be dropped from candidates),
// and an error only for unrecoverable failures (I/O, cancel).
func (e *Engine) buildMeasuredLossTable(ctx context.Context, bank *core.TensorBank,
	cands []core.DType, imatrix map[string]profile.ImatrixStats,
	existing map[string]map[core.DType]float64,
	save func(map[string]map[core.DType]float64) error,
) (map[string]map[core.DType]float64, []core.DType, error) {

	table := make(map[string]map[core.DType]float64, len(existing))
	for name, m := range existing {
		table[name] = make(map[core.DType]float64, len(m))
		for d, v := range m {
			table[name][d] = v
		}
	}

	srcPath := e.payloadSource()
	tensors := e.measuredEligibleTensors(bank, srcPath)
	if len(tensors) == 0 {
		return table, nil, nil
	}

	// Sampling fraction and per-tensor keep sets.
	var totalElems uint64
	for _, mt := range tensors {
		totalElems += mt.desc.Elements
	}
	sampleBudget := uint64(measuredSampleElements)
	for {
		f := 1.0
		if totalElems > sampleBudget {
			f = float64(sampleBudget) / float64(totalElems)
		}
		e.applySampleRows(tensors, f)
		need := e.estimateMeasureScratch(bank, tensors, cands)
		if free, ok := tensorbank.DiskFree(e.workDir()); ok && free < uint64(float64(need)*1.2) {
			if sampleBudget <= 32<<20 {
				break // floor reached; proceed and hope
			}
			sampleBudget /= 2
			continue
		}
		break
	}

	measureDir := filepath.Join(e.workDir(), "measure")
	if err := os.MkdirAll(measureDir, 0o755); err != nil {
		return table, nil, err
	}
	defer os.RemoveAll(measureDir)

	// Write the sampled GGUF.
	picks := make([]tensorbank.RowSample, len(tensors))
	for i, mt := range tensors {
		picks[i] = tensorbank.RowSample{Name: mt.desc.Name, Keep: mt.keep}
	}
	samplePath := filepath.Join(measureDir, "sample.gguf")
	e.obsProgress(core.StageSolve, 0, "measure: sampling rows")
	if err := tensorbank.SampleRows(ctx, srcPath, samplePath, picks, nil); err != nil {
		return table, nil, fmt.Errorf("measure: sample rows: %w", err)
	}

	capsQ, err := e.caps(ctx, orchestrate.ToolLlamaQuantize)
	if err != nil {
		return table, nil, err
	}
	canMeasure := capsQ.Has("--allow-requantize") && (len(capsQ.Types) == 0 || capsQ.HasType("F32"))
	var unmeasured []core.DType

	// Determine which dtypes still need measurement.
	for _, d := range cands {
		if e.allMeasured(tensors, table, d) {
			continue
		}
		if !canMeasure {
			e.fallbackOrDrop(d, tensors, table, imatrix, &unmeasured)
			continue
		}
		if err := ctx.Err(); err != nil {
			return table, unmeasured, err
		}
		keepSet := e.measureKeepSet(tensors, d, imatrix)
		if len(keepSet) == 0 {
			continue
		}
		e.obsProgress(core.StageSolve, 0, fmt.Sprintf("measure: %s", d))
		ok := e.measureOneDType(ctx, d, samplePath, measureDir, keepSet, tensors, imatrix, table, capsQ)
		if !ok {
			e.fallbackOrDrop(d, tensors, table, imatrix, &unmeasured)
			continue
		}
		if save != nil {
			if err := save(table); err != nil {
				return table, unmeasured, err
			}
		}
		e.printf("  measured %-8s %d tensors\n", d, len(keepSet))
	}
	return table, unmeasured, nil
}

// measuredEligibleTensors filters the bank to tensors the exact table
// covers: quantizable, float source, present in the payload file.
func (e *Engine) measuredEligibleTensors(bank *core.TensorBank, srcPath string) []measuredTensor {
	var out []measuredTensor
	for _, t := range bank.Tensors {
		if !t.Quantizable() || !t.DType.IsFloat() {
			continue
		}
		if len(t.Shape) < 2 {
			continue
		}
		R := t.Shape[1]
		S := uint64(1)
		for _, d := range t.Shape[2:] {
			S *= d
		}
		out = append(out, measuredTensor{desc: t, R: R, S: S})
	}
	return out
}

// applySampleRows computes deterministic row keep-sets for each tensor.
func (e *Engine) applySampleRows(tensors []measuredTensor, f float64) {
	for i := range tensors {
		mt := &tensors[i]
		R := mt.R
		minRows := uint64(32)
		if len(mt.desc.Shape) >= 3 {
			minRows = 8
		}
		keep := uint64(math.Ceil(f * float64(R)))
		if keep < minRows && R >= minRows {
			keep = minRows
		}
		if keep > R {
			keep = R
		}
		if keep < 1 {
			keep = 1
		}
		mt.keep = make([]uint64, keep)
		for j := uint64(0); j < keep; j++ {
			mt.keep[j] = uint64(math.Floor((float64(j) + 0.5) * float64(R) / float64(keep)))
			if mt.keep[j] >= R {
				mt.keep[j] = R - 1
			}
		}
	}
}

// estimateMeasureScratch estimates peak scratch: sample + trimmed sample +
// quantized + dequantized ≈ 2.5× sample size at the current budget.
func (e *Engine) estimateMeasureScratch(bank *core.TensorBank, tensors []measuredTensor, cands []core.DType) uint64 {
	var sampleBytes uint64
	for _, mt := range tensors {
		es := uint64(4)
		if mt.desc.DType == core.DTypeF16 || mt.desc.DType == core.DTypeBF16 {
			es = 2
		}
		keep := uint64(len(mt.keep))
		sampleBytes += mt.desc.Shape[0] * keep * mt.S * es
	}
	return sampleBytes * 5 / 2 // sample + trim + q + f32 + headroom
}

// allMeasured reports whether every eligible tensor already has a table
// entry for d.
func (e *Engine) allMeasured(tensors []measuredTensor, table map[string]map[core.DType]float64, d core.DType) bool {
	for _, mt := range tensors {
		if _, ok := table[mt.desc.Name][d]; !ok {
			return false
		}
	}
	return true
}

// measureKeepSet returns the sampled tensors eligible for dtype d:
// block-aligned ne0 and an imatrix entry when d requires one.
func (e *Engine) measureKeepSet(tensors []measuredTensor, d core.DType, imatrix map[string]profile.ImatrixStats) map[string]struct{} {
	g, ok := d.BaseTensorType().Geometry()
	if !ok {
		return nil
	}
	out := map[string]struct{}{}
	for _, mt := range tensors {
		if mt.desc.Shape[0]%g.BlockSize != 0 {
			continue
		}
		if d.RequiresImatrix() {
			if _, hit := profile.LookupImatrix(imatrix, mt.desc.Name); !hit {
				continue
			}
		}
		out[mt.desc.Name] = struct{}{}
	}
	return out
}

// fallbackOrDrop computes d from the Go table when its codec is exact, or
// adds d to unmeasured (which the caller must drop from candidates).
func (e *Engine) fallbackOrDrop(d core.DType, tensors []measuredTensor,
	table map[string]map[core.DType]float64, imatrix map[string]profile.ImatrixStats,
	unmeasured *[]core.DType) {
	if !goFallbackDTypes[d.BaseTensorType()] {
		*unmeasured = append(*unmeasured, d)
		e.printf("  measure: %s unmeasured; dropped from candidates (Go proxy unreliable)\n", d)
		return
	}
	// Compute with the Go table for the missing entries only.
	bank := e.Run.Bank
	if fp := e.payloadSource(); fp != bank.SourcePath {
		sb := *bank
		sb.SourcePath = fp
		bank = &sb
	}
	goTable, err := profile.BuildExactLossTableCfg(bank, []core.DType{d}, imatrix, nil, profile.ExactConfig{
		Context: context.Background(),
	})
	if err != nil {
		*unmeasured = append(*unmeasured, d)
		e.printf("  measure: %s Go fallback failed (%v); dropped\n", d, err)
		return
	}
	for name, m := range goTable {
		if table[name] == nil {
			table[name] = map[core.DType]float64{}
		}
		if v, ok := m[d]; ok {
			if _, exists := table[name][d]; !exists {
				table[name][d] = v
			}
		}
	}
	e.printf("  measured %-8s (Go fallback) %d tensors\n", d, len(goTable))
}

// measureOneDType runs quantize → verify → dequantize → compare for one
// dtype and stores the extrapolated wSSE into table. Returns false on any
// tool or verification failure.
func (e *Engine) measureOneDType(ctx context.Context, d core.DType,
	samplePath, measureDir string, keepSet map[string]struct{},
	tensors []measuredTensor, imatrix map[string]profile.ImatrixStats,
	table map[string]map[core.DType]float64, capsQ *orchestrate.Capabilities) bool {

	// Trim sample to the keep-set if needed.
	srcForD := samplePath
	allNames := map[string]struct{}{}
	for _, mt := range tensors {
		allNames[mt.desc.Name] = struct{}{}
	}
	needTrim := len(keepSet) < len(allNames)
	trimPath := filepath.Join(measureDir, fmt.Sprintf("sample-%s.gguf", d))
	qPath := filepath.Join(measureDir, fmt.Sprintf("q-%s.gguf", d))
	fPath := filepath.Join(measureDir, fmt.Sprintf("f-%s.gguf", d))
	defer os.Remove(trimPath)
	defer os.Remove(qPath)
	defer os.Remove(fPath)

	if needTrim {
		if err := tensorbank.Trim(ctx, samplePath, keepSet, trimPath, nil); err != nil {
			e.printf("  measure: %s trim failed: %v\n", d, err)
			return false
		}
		srcForD = trimPath
	}

	// Quantize.
	qReq := orchestrate.QuantizeRequest{
		ProfileID:  "measure-" + string(d),
		SourcePath: srcForD,
		OutputPath: qPath,
		Type:       d,
		Pure:       true,
	}
	e.fillQuantizeRequest(&qReq)
	iv, err := orchestrate.PlanQuantize(qReq, capsQ, e.Run.Config.Tools.LlamaQuantize)
	if err != nil {
		e.printf("  measure: %s plan failed: %v\n", d, err)
		return false
	}
	if _, err := runOK(ctx, e.Runner, iv); err != nil {
		e.printf("  measure: %s quantize failed: %v\n", d, err)
		return false
	}
	if err := e.verifyAnchor(qPath, d); err != nil {
		e.printf("  measure: %s verify failed: %v\n", d, err)
		return false
	}

	// Dequantize.
	dReq := orchestrate.QuantizeRequest{
		ProfileID:       "measure-deq-" + string(d),
		SourcePath:      qPath,
		OutputPath:      fPath,
		Type:            core.DTypeF32,
		AllowRequantize: true,
		Dequantize:      true,
		SourceQuantized: true,
		Pure:            true,
	}
	iv, err = orchestrate.PlanQuantize(dReq, capsQ, e.Run.Config.Tools.LlamaQuantize)
	if err != nil {
		e.printf("  measure: %s dequant plan failed: %v\n", d, err)
		return false
	}
	if _, err := runOK(ctx, e.Runner, iv); err != nil {
		e.printf("  measure: %s dequant failed: %v\n", d, err)
		return false
	}

	// Compare.
	if err := e.measureCompare(ctx, samplePath, fPath, keepSet, tensors, imatrix, table, d); err != nil {
		e.printf("  measure: %s compare failed: %v\n", d, err)
		return false
	}
	return true
}

// measureCompare computes the extrapolated wSSE for each tensor in the
// keep-set by comparing the sampled source rows against the F32
// dequantized rows.
func (e *Engine) measureCompare(ctx context.Context, samplePath, fPath string,
	keepSet map[string]struct{}, tensors []measuredTensor,
	imatrix map[string]profile.ImatrixStats,
	table map[string]map[core.DType]float64, d core.DType) error {

	ss, err := tensorbank.OpenSource(samplePath)
	if err != nil {
		return err
	}
	defer ss.Close()
	sf, err := tensorbank.Parse(ss)
	if err != nil {
		return err
	}
	fs, err := tensorbank.OpenSource(fPath)
	if err != nil {
		return err
	}
	defer fs.Close()
	ff, err := tensorbank.Parse(fs)
	if err != nil {
		return err
	}

	for _, mt := range tensors {
		if err := ctx.Err(); err != nil {
			return err
		}
		if _, ok := keepSet[mt.desc.Name]; !ok {
			continue
		}
		sti, ok := sf.FindTensor(mt.desc.Name)
		if !ok {
			continue
		}
		fti, ok := ff.FindTensor(mt.desc.Name)
		if !ok {
			continue
		}
		if fti.DType != core.DTypeF32 {
			return fmt.Errorf("dequantized %s is %s, want F32", mt.desc.Name, fti.DType)
		}
		if sti.Shape[0] != fti.Shape[0] || sti.Elements != fti.Elements {
			return fmt.Errorf("shape mismatch on %s", mt.desc.Name)
		}

		ne0 := sti.Shape[0]
		keep := uint64(len(mt.keep))
		nEl := ne0 * keep * mt.S

		srcVals := make([]float32, nEl)
		dstVals := make([]float32, nEl)
		bufSize := nEl * 4
		if sti.DType == core.DTypeF16 || sti.DType == core.DTypeBF16 {
			bufSize = nEl * 2
		}
		buf := make([]byte, bufSize)
		if _, err := ss.ReadAt(buf, sf.PayloadOffset(sti)); err != nil {
			return err
		}
		decodeMeasuredFloats(srcVals, buf, sti.DType)

		buf2 := make([]byte, nEl*4)
		if _, err := fs.ReadAt(buf2, ff.PayloadOffset(fti)); err != nil {
			return err
		}
		decodeMeasuredFloats(dstVals, buf2, core.DTypeF32)

		// Importance via the solver's joined stats on the sampled rows.
		imp := make([]float32, nEl)
		for i := range imp {
			imp[i] = 1
		}
		if st, hit := profile.LookupImatrix(imatrix, mt.desc.Name); hit && len(st.Values) > 0 {
			layout, ok := profile.LayoutFor(st.Values, ne0, keep*mt.S)
			if ok {
				layout.Fill(imp, st.Values, 0, keep*mt.S)
			}
		}

		var wSSE float64
		for i := uint64(0); i < nEl; i++ {
			e := float64(srcVals[i]) - float64(dstVals[i])
			wSSE += float64(imp[i]) * e * e
		}
		// Extrapolate from sampled rows to the full tensor.
		extrapolation := float64(mt.R) / float64(keep)
		wSSE *= extrapolation

		if math.IsNaN(wSSE) || math.IsInf(wSSE, 0) {
			continue
		}
		if table[mt.desc.Name] == nil {
			table[mt.desc.Name] = map[core.DType]float64{}
		}
		table[mt.desc.Name][d] = wSSE
	}
	return nil
}

// decodeMeasuredFloats decodes a raw payload buffer into float32 values.
func decodeMeasuredFloats(dst []float32, buf []byte, d core.DType) {
	switch d {
	case core.DTypeF32:
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
	case core.DTypeF16:
		for i := range dst {
			dst[i] = qtype.F16ToF32(binary.LittleEndian.Uint16(buf[i*2:]))
		}
	case core.DTypeBF16:
		for i := range dst {
			dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(buf[i*2:])) << 16)
		}
	}
}
