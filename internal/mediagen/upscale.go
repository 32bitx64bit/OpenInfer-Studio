package mediagen

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
)

type UpscaleParams struct {
	Image        string `json:"image,omitempty"`
	ImagePath    string `json:"image_path,omitempty"`
	Upscaler     string `json:"upscaler,omitempty"`
	Repeats      int    `json:"repeats,omitempty"`
	TileSize     int    `json:"tile_size,omitempty"`
	OutputFormat string `json:"output_format,omitempty"`
}

// UpscaleStage is the optional workflow seam for native model upscaling.
// Outputs use managed relative paths and ordinary image Job rows. The native
// endpoint is synchronous, has no upstream job ID/cancel endpoint, and may
// continue GPU work after its HTTP request is canceled. Cancellation bounds
// this call and prevents storing results; it does not promise GPU abort.
func (m *Manager) UpscaleStage(ctx context.Context, modelID string, p UpscaleParams) (*Job, error) {
	return m.RunUpscale(ctx, modelID, p, nil)
}

// RunUpscale additionally permits launch overrides such as ESRGAN. A runtime
// must advertise upscale:true; named models also need image_upscale:true.
func (m *Manager) RunUpscale(ctx context.Context, modelID string, p UpscaleParams, ov *LoadOverrides) (*Job, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if (p.Image == "") == (p.ImagePath == "") {
		return nil, fmt.Errorf("provide exactly one of image or image_path")
	}
	if p.Repeats == 0 {
		p.Repeats = 1
	}
	if ov != nil && ov.ESRGAN != "" && p.Upscaler == "" {
		p.Upscaler = strings.TrimSuffix(filepath.Base(ov.ESRGAN), filepath.Ext(ov.ESRGAN))
	}
	if strings.TrimSpace(p.Upscaler) == "" {
		return nil, fmt.Errorf("select an advertised RGB model upscaler")
	}
	if p.Repeats < 1 || p.Repeats > 4 {
		return nil, fmt.Errorf("upscale repeats must be 1..4")
	}
	if p.TileSize < 0 || p.TileSize > 2048 {
		return nil, fmt.Errorf("upscale tile_size must be 0..2048")
	}
	if p.OutputFormat != "" && !containsChoice([]string{"png", "jpeg", "webp"}, p.OutputFormat) {
		return nil, fmt.Errorf("unsupported upscale output_format")
	}
	input := imageInput{field: "image", inline: p.Image, path: p.ImagePath}
	if _, _, err := readImageInput(input); err != nil {
		return nil, err
	}
	port, err := m.stageServer(ctx, modelID, ov)
	if err != nil {
		return nil, err
	}
	caps, err := m.generationCapabilities(ctx, modelID)
	if err != nil {
		return nil, err
	}
	if !caps.Known || !caps.Upscale {
		return nil, fmt.Errorf("loaded-model API does not advertise native RGB model upscaling: %s", caps.Reason)
	}
	if p.Upscaler != "" {
		allowed := false
		for _, up := range caps.Upscalers {
			if up.Name == p.Upscaler && up.Model && up.ImageUpscale {
				allowed = true
				break
			}
		}
		if !allowed {
			return nil, fmt.Errorf("upscaler %q is not advertised as a compatible RGB model", p.Upscaler)
		}
	}
	raw, mime, err := readImageInput(input) // revalidate after a potentially lengthy model load
	if err != nil {
		return nil, err
	}
	w, h, _, err := imageInfo(raw)
	if err != nil {
		return nil, err
	}
	// Upstream ESRGAN scale is model-specific and not advertised in this
	// catalog. Never assume 4x: enforce input/local limits here and let the
	// endpoint enforce its advertised final output limit before GPU work.
	if (caps.Limits.MaxUpscaleWidth > 0 && w > caps.Limits.MaxUpscaleWidth) || (caps.Limits.MaxUpscaleHeight > 0 && h > caps.Limits.MaxUpscaleHeight) {
		return nil, fmt.Errorf("image exceeds advertised upscale dimensions")
	}
	if !m.serverMatchesPort(modelID, port) {
		return nil, ErrServerNotRunning
	}
	id := uuid.NewString()
	params, _ := json.Marshal(p)
	ts := now()
	if _, err = m.db.Exec(`INSERT INTO media_jobs(id,model_id,kind,state,prompt,params_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)`, id, modelID, KindImage, StateRunning, "", string(params), ts, ts); err != nil {
		return nil, err
	}
	jctx, cancel := context.WithCancelCause(ctx)
	m.mu.Lock()
	m.jobs[id] = &jobHandle{cancel: cancel, modelID: modelID, port: port}
	m.mu.Unlock()
	defer func() { cancel(nil); m.mu.Lock(); delete(m.jobs, id); m.mu.Unlock() }()
	runtimeID, logPath, pid := m.serverMeta(modelID)
	_, _ = m.db.Exec(`UPDATE media_jobs SET runtime_id=?,log_path=?,pid=? WHERE id=?`, runtimeID, logPath, pid, id)
	m.publish("media.progress", map[string]any{"id": id, "model_id": modelID, "state": StateRunning, "message": "Upscaling image…"})
	finishErr := func(err error) (*Job, error) {
		if !jobCanceledByCrash(jctx) {
			state := StateFailed
			if jctx.Err() != nil {
				state = StateCanceled
				err = jctx.Err()
			}
			_ = m.setState(id, state, "", err.Error())
			m.publish("media.progress", map[string]any{"id": id, "model_id": modelID, "state": state, "error": err.Error()})
		}
		j, _ := m.Get(id)
		return j, err
	}
	body := map[string]any{"image": "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(raw), "repeats": p.Repeats}
	if p.Upscaler != "" {
		body["upscaler"] = p.Upscaler
	}
	if p.TileSize > 0 {
		body["tile_size"] = p.TileSize
	}
	if p.OutputFormat != "" {
		body["output_format"] = p.OutputFormat
	}
	payload, _ := json.Marshal(body)
	requestCtx, requestCancel := context.WithTimeout(jctx, 5*time.Minute)
	defer requestCancel()
	req, _ := http.NewRequestWithContext(requestCtx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/sdcpp/v1/upscale", port), bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	resp, err := m.doSD(req, port)
	if err != nil {
		return finishErr(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return finishErr(fmt.Errorf("sd-server upscale HTTP %d", resp.StatusCode))
	}
	var result map[string]any
	if err = readSDJSON(resp.Body, 128<<20, &result); err != nil {
		return finishErr(err)
	}
	if jctx.Err() != nil {
		return finishErr(jctx.Err())
	}
	if !m.serverMatchesPort(modelID, port) {
		return finishErr(ErrServerNotRunning)
	}
	outWidth, _ := result["width"].(float64)
	outHeight, _ := result["height"].(float64)
	maxW, maxH := 8192, 8192
	if caps.Limits.MaxUpscaleWidth > 0 && caps.Limits.MaxUpscaleWidth < maxW {
		maxW = caps.Limits.MaxUpscaleWidth
	}
	if caps.Limits.MaxUpscaleHeight > 0 && caps.Limits.MaxUpscaleHeight < maxH {
		maxH = caps.Limits.MaxUpscaleHeight
	}
	if outWidth < 1 || outHeight < 1 || outWidth > float64(maxW) || outHeight > float64(maxH) {
		return finishErr(errors.New("sd-server upscale returned invalid output dimensions"))
	}
	paths, metadata, err := m.storeSDResult(id, modelID, KindImage, GenerateParams{OutputFormat: p.OutputFormat}, map[string]any{"result": result})
	if err != nil {
		return finishErr(err)
	}
	if _, err = m.db.Exec(`UPDATE media_jobs SET state=?,output_path=?,result_json=?,width=?,height=?,finished_at=?,updated_at=? WHERE id=?`, StateComplete, firstPath(paths), string(metadata), int(outWidth), int(outHeight), now(), now(), id); err != nil {
		return finishErr(err)
	}
	m.publish("media.progress", map[string]any{"id": id, "model_id": modelID, "state": StateComplete, "outputs": paths, "message": "Complete"})
	return m.Get(id)
}

// stageServer shares the existing managed load/reload lifecycle. Individual
// stage cancellation stops waiting without canceling a load shared by other
// callers. The load itself retains the manager's startup deadline.
func (m *Manager) stageServer(ctx context.Context, modelID string, ov *LoadOverrides) (int, error) {
	if ctx.Err() != nil {
		return 0, ctx.Err()
	}
	if port, ok := m.ServerPort(modelID); ok && (ov == nil || m.serverSatisfies(modelID, *ov)) {
		return port, nil
	}
	if _, ok := m.ServerPort(modelID); ok {
		m.StopServer(modelID)
	}
	settings := DefaultLoadSettings()
	if m.lib != nil {
		if raw, ok := m.lib.LastGood(modelID); ok {
			_ = json.Unmarshal(raw, &settings)
		}
	}
	if ov != nil && !ov.IsZero() {
		ov.Apply(&settings)
		settings.SaveOnSuccess = false
	}
	type startResult struct {
		port int
		err  error
	}
	ch := make(chan startResult, 1)
	go func() { port, err := m.EnsureServer(modelID, settings); ch <- startResult{port, err} }()
	select {
	case r := <-ch:
		return r.port, r.err
	case <-ctx.Done():
		return 0, ctx.Err()
	}
}
