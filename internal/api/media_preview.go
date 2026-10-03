package api

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/openinfer/openinfer-studio/internal/instances"
	"github.com/openinfer/openinfer-studio/internal/mediagen"
	"github.com/openinfer/openinfer-studio/internal/models"
)

// previewDiffusionLoad shows the sd-server command for a checkpoint instead
// of the llama-server argv. The load dialog branches on modality and calls
// the media server endpoint for diffusion rows. The response also carries a
// pipeline-component report (VAE / text encoders / tokenizer): ready ones
// are paired into the command, missing ones are listed with their source
// repo paths and sizes so the dialog can offer a download.
//
// A missing/unresolvable sd.cpp runtime is not an error here: it comes back
// 200 with a warning (and no args/command) so the dialog can still render
// the component report and point the user at the Runtimes page, instead of
// a failed fetch.
func (h *handlers) previewDiffusionLoad(w http.ResponseWriter, r *http.Request, m *models.Model) {
	var s mediagen.LoadSettings
	if r.Body != nil && r.ContentLength != 0 {
		raw := map[string]json.RawMessage{}
		if !decodeJSON(w, r, &raw) {
			return
		}
		merged, _ := json.Marshal(mediagen.DefaultLoadSettings())
		base := map[string]json.RawMessage{}
		_ = json.Unmarshal(merged, &base)
		for k, v := range raw {
			base[k] = v
		}
		final, _ := json.Marshal(base)
		if err := json.Unmarshal(final, &s); err != nil {
			writeErr(w, 400, "invalid diffusion load settings", err)
			return
		}
	}

	repoID := h.diffusionSourceRepo(m)
	report := h.componentReport(r.Context(), m, repoID)
	missing := mediagen.MissingCompanions(report)
	var missingBytes int64
	for _, c := range missing {
		missingBytes += c.Size
	}
	var warnings []string
	if len(missing) > 0 {
		var names []string
		for _, c := range missing {
			names = append(names, c.Label)
		}
		warnings = append(warnings, "Pipeline incomplete — missing "+strings.Join(names, ", ")+
			". Download them below, or the server will refuse to start.")
	}

	resp := map[string]any{
		"modality":                 "diffusion",
		"can_load":                 false,
		"args":                     []string{},
		"command":                  "",
		"source_repo":              repoID,
		"components":               report,
		"components_missing":       len(missing),
		"components_missing_bytes": missingBytes,
		"resolutions":              []map[string]any{},
	}

	rt, err := h.d.Media.ResolveRuntime(m.ID, s.RuntimeID)
	if err != nil {
		warnings = append(warnings, "No stable-diffusion.cpp runtime available: "+err.Error()+
			". Install one from the Runtimes page to load this model.")
		resp["warnings"] = warnings
		writeJSON(w, 200, resp)
		return
	}

	// Same helper the launcher uses (PrepareLaunch), so the preview shows
	// exactly what will run: resolved sd-server path (not rt.ExecutablePath),
	// auto-paired companions, and the real --model/--diffusion-model choice.
	help, _ := h.d.RT.HelpOutput(rt.ID)
	exe, args, resolved, _, _, prepWarnings, err := mediagen.PrepareLaunch(rt, help, m.PrimaryPath, s)
	warnings = append(warnings, prepWarnings...)
	if err != nil {
		// Keep the component report visible when preflight fails. A bare 400
		// used to leave the dialog showing its previous successful preview.
		resp["warnings"] = append(warnings, err.Error())
		writeJSON(w, 200, resp)
		return
	}

	root := mediagen.ModelRoot(m.PrimaryPath)
	componentOnly := mediagen.IsComponentDiffusionFile(m.PrimaryPath)
	modelFlagResolved := "--model (full checkpoint)"
	if componentOnly {
		modelFlagResolved = "--diffusion-model (transformer-only)"
	}
	resolutions := []map[string]any{
		{"setting": "Runtime", "auto": s.RuntimeID, "resolved": rt.ID + " (" + rt.Backend + ")"},
		{"setting": "Engine", "auto": "sd.cpp", "resolved": "sd-server"},
		{"setting": "Model flag", "auto": "auto-detected", "resolved": modelFlagResolved},
	}
	for _, d := range []struct{ label, before, after string }{
		{"VAE", s.VAE, resolved.VAE},
		{"TAESD", s.TAESD, resolved.TAESD},
		{"LLM text encoder", s.LLM, resolved.LLM},
		{"LLM vision encoder", s.LLMVision, resolved.LLMVision},
		{"T5-XXL text encoder", s.T5XXL, resolved.T5XXL},
		{"CLIP-L text encoder", s.ClipL, resolved.ClipL},
		{"CLIP-G text encoder", s.ClipG, resolved.ClipG},
		{"CLIP vision encoder", s.ClipVision, resolved.ClipVision},
		{"Tokenizer", s.Tokenizer, resolved.Tokenizer},
		{"Upscaler (ESRGAN)", s.ESRGAN, resolved.ESRGAN},
		{"ControlNet", s.ControlNet, resolved.ControlNet},
	} {
		if d.before == "" && d.after != "" {
			resolutions = append(resolutions, map[string]any{
				"setting": d.label, "auto": "auto", "resolved": "auto → " + displayRelPath(root, d.after),
			})
		}
	}

	resp["args"] = args
	resp["environment"] = mediagen.SDLaunchEnvironment(exe, args, rt.Backend)
	resp["can_load"] = true
	resp["command"] = strings.Join(append([]string{exe}, args...), " ")
	resp["resolutions"] = resolutions
	resp["warnings"] = warnings
	writeJSON(w, 200, resp)
}

// displayRelPath shortens a companion path to be relative to the model's
// managed root (…/components/vae/x.safetensors instead of the full
// filesystem path) when it lives under that root; otherwise the full path
// is returned unchanged.
func displayRelPath(root, path string) string {
	if root == "" || path == "" {
		return path
	}
	rel, err := filepath.Rel(root, path)
	if err != nil || strings.HasPrefix(rel, "..") {
		return path
	}
	return rel
}

// diffusionSourceRepo resolves the Hugging Face id a checkpoint came from:
// the stamped library value first, then the managed dir name (owner--repo).
func (h *handlers) diffusionSourceRepo(m *models.Model) string {
	repo := strings.TrimSpace(m.SourceRepo)
	if repo != "" && !strings.HasPrefix(repo, "local/") {
		return repo
	}
	return mediagen.SourceRepoFor(h.d.Layout.Models, m.PrimaryPath)
}

// componentReport merges local companion discovery with the source repo's
// file list (vae/, text_encoders/, …) to show ready/missing pipeline parts.
func (h *handlers) componentReport(ctx context.Context, m *models.Model, repoID string) []mediagen.Companion {
	root := mediagen.ModelRoot(m.PrimaryPath)
	if repoID == "" || h.d.HF == nil {
		return mediagen.ComponentReport(root, m.PrimaryPath, nil, nil)
	}
	info, err := h.d.HF.Repo(ctx, repoID)
	if err != nil || info == nil {
		return mediagen.ComponentReport(root, m.PrimaryPath, nil, nil)
	}
	var paths []string
	sizes := map[string]int64{}
	for _, f := range info.Files {
		if f.Path == "" {
			continue
		}
		paths = append(paths, f.Path)
		sizes[f.Path] = f.Size
	}
	return mediagen.ComponentReport(root, m.PrimaryPath, paths, sizes)
}

// sdWeightBytesPerElem is a rough bytes-per-weight table for sd-server's
// --type values, used only to scale the estimate when the user overrides
// the weight type. Diffusion checkpoints are not GGUF tensor tables the way
// llama.cpp models are (safetensors, mixed dtypes per block), so this is a
// coarse approximation, not a real quantizer — always presented as such.
var sdWeightBytesPerElem = map[string]float64{
	"f32": 4.0, "fp32": 4.0,
	"f16": 2.0, "fp16": 2.0, "bf16": 2.0,
	"q8_0": 1.06,
	"q6_k": 0.82,
	"q5_k": 0.69, "q5_0": 0.69, "q5_1": 0.75,
	"q4_k": 0.56, "q4_0": 0.56, "q4_1": 0.63,
	"q3_k": 0.44,
	"q2_k": 0.35,
}

// sdWeightScale returns a rough size multiplier for an sd-server --type
// override, assuming source checkpoint files are authored at ~f16 (the
// common safetensors distribution format). "" (unset) is 1.0 — no change.
func sdWeightScale(weightType string) float64 {
	if weightType == "" {
		return 1.0
	}
	if bpw, ok := sdWeightBytesPerElem[strings.ToLower(weightType)]; ok {
		return bpw / 2.0
	}
	return 1.0
}

// headroomBytes reserves 10% of a budget, matching the placement headroom
// instances.EstimateMemory uses for the LLM estimator's GPU/RAM bars.
func headroomBytes(budget uint64) uint64 {
	if budget == 0 {
		return 0
	}
	return budget - budget/10
}

// estimateDiffusionLoad projects VRAM/RAM placement for a diffusion
// pipeline. Returns the exact instances.Estimate shape the LLM estimator
// uses (fits/fits_gpu/fits_cpu, gpu_bytes/cpu_bytes, budgets, …) so the load
// dialog can reuse its GPU/RAM bars unchanged.
//
// Diffusion has no KV cache; instead this models sd.cpp's --auto-fit
// placement roughly: the diffusion (DiT/UNet) weights plus a compute
// reserve are placed on the compute GPU first; the rest of the pipeline
// (text encoders + VAE + small extras) is placed on GPU only if it fits
// in what's left, otherwise the whole remainder goes to system RAM as one
// bundle — matching the real sd.cpp log this was modeled on (a 22 GB
// Qwen-Image pipeline: 5.6 GB DiT → GPU, 16.7 GB conditioner + 0.6 GB VAE →
// RAM, because the 17.3 GB remainder didn't fit after the DiT). This is a
// rough approximation, not sd.cpp's real graph-cut algorithm, and is
// labeled as such in "note".
func (h *handlers) estimateDiffusionLoad(w http.ResponseWriter, r *http.Request, m *models.Model) {
	var s mediagen.LoadSettings
	if r.Body != nil && r.ContentLength != 0 {
		raw := map[string]json.RawMessage{}
		if decodeJSON(w, r, &raw) {
			merged, _ := json.Marshal(mediagen.DefaultLoadSettings())
			base := map[string]json.RawMessage{}
			_ = json.Unmarshal(merged, &base)
			for k, v := range raw {
				base[k] = v
			}
			final, _ := json.Marshal(base)
			_ = json.Unmarshal(final, &s)
		} else {
			return
		}
	}

	primary := m.SizeBytes
	if primary <= 0 {
		if st, err := os.Stat(m.PrimaryPath); err == nil {
			primary = st.Size()
		}
	}

	root := mediagen.ModelRoot(m.PrimaryPath)
	companions := mediagen.DiscoverCompanions(root, m.PrimaryPath)
	sizeOf := func(role string) int64 {
		p, ok := companions[role]
		if !ok {
			return 0
		}
		if st, err := os.Stat(p); err == nil {
			return st.Size()
		}
		return 0
	}
	// Conditioner = text/vision encoders (the bundle sd.cpp calls out
	// separately from the DiT/UNet in its placement log).
	conditioner := sizeOf(mediagen.CompanionLLM) + sizeOf(mediagen.CompanionT5XXL) +
		sizeOf(mediagen.CompanionClipL) + sizeOf(mediagen.CompanionClipG) + sizeOf(mediagen.CompanionClipVision)
	vae := sizeOf(mediagen.CompanionVAE) + sizeOf(mediagen.CompanionTAESD)
	other := sizeOf(mediagen.CompanionUpscaler) + sizeOf(mediagen.CompanionControlNet)

	scale := sdWeightScale(s.WeightType)
	primary = int64(float64(primary) * scale)
	conditioner = int64(float64(conditioner) * scale)
	vae = int64(float64(vae) * scale)
	other = int64(float64(other) * scale)

	// Compute/activation scratch: scales with checkpoint size, same shape as
	// the original heuristic (resolution-dependent in spirit; sd.cpp's real
	// scratch depends on width/height/batch, which this endpoint does not
	// see per-generation).
	compute := int64(1536 << 20)
	if primary > 6<<30 {
		compute = 4 << 30
	} else if primary > 3<<30 {
		compute = 3 << 30
	}
	overhead := int64(128 << 20)

	weightsTotal := primary + conditioner + vae + other
	total := weightsTotal + compute + overhead

	hw := h.hardwareInfo()
	var vram uint64
	for _, g := range hw.GPUs {
		vram += g.VRAM
	}
	ramBudget := hw.RAMAvailable

	est := instances.Estimate{
		WeightsBytes:  weightsTotal,
		ComputeBytes:  compute,
		OverheadBytes: overhead,
		TotalBytes:    total,
	}

	offloadToCPU := s.OffloadCPU
	unified := vram == 0
	switch {
	case offloadToCPU || unified:
		est.GPUBytes = 0
		est.CPUBytes = total
		est.GPUBudgetBytes = 0
		est.CPUBudgetBytes = ramBudget
		if unified && !offloadToCPU {
			est.BudgetKind = "unified RAM"
		} else {
			est.BudgetKind = "RAM"
		}
		est.BudgetBytes = ramBudget
		est.FitsGPU = true
		est.FitsCPU = est.CPUBytes <= int64(headroomBytes(ramBudget))
		est.Fits = est.FitsCPU
	default:
		est.GPUBudgetBytes = vram
		est.CPUBudgetBytes = ramBudget
		est.BudgetKind = "VRAM+RAM"
		est.BudgetBytes = vram
		primaryPlusCompute := primary + compute + overhead
		gpuHeadroom := int64(headroomBytes(vram))
		if primaryPlusCompute <= gpuHeadroom {
			est.GPUBytes = primaryPlusCompute
			remaining := gpuHeadroom - primaryPlusCompute
			bundle := conditioner + vae + other
			if bundle <= remaining {
				est.GPUBytes += bundle
				est.CPUBytes = 0
			} else {
				est.CPUBytes = bundle
			}
		} else {
			// Even the diffusion weights alone don't fit the GPU budget;
			// sd.cpp's auto-fit would fall further back (disk/CPU graph
			// segmentation). Roughly model that as a full RAM fallback.
			est.GPUBytes = 0
			est.CPUBytes = total
		}
		est.FitsGPU = est.GPUBytes <= gpuHeadroom
		est.FitsCPU = est.CPUBytes <= int64(headroomBytes(ramBudget))
		est.Fits = est.FitsGPU && est.FitsCPU
	}

	note := "rough estimate: models sd.cpp's --auto-fit placement (diffusion weights + compute reserve on GPU first, " +
		"text encoders/VAE on GPU only if they fit, otherwise all in RAM); no KV cache"
	if scale != 1.0 {
		note += "; sizes scaled for --type " + s.WeightType + " (bytes-per-weight approximation, not exact)"
	}
	est.Note = note

	// Marshal through instances.Estimate's own json tags so the diffusion
	// estimate is byte-for-byte the same shape as the LLM estimate (bug: it
	// used to be a bespoke {weights_bytes, scratch_bytes, gpu_budget_bytes,
	// fits_gpu} shape the dialog's GPU/RAM bars couldn't parse). A few extra
	// diffusion-only fields ride alongside for the Image Studio breakdown.
	out := map[string]any{}
	if b, err := json.Marshal(est); err == nil {
		_ = json.Unmarshal(b, &out)
	}
	out["modality"] = "diffusion"
	out["diffusion_weights_bytes"] = primary
	out["conditioner_bytes"] = conditioner
	out["vae_bytes"] = vae
	out["other_companion_bytes"] = other
	writeJSON(w, 200, out)
}
