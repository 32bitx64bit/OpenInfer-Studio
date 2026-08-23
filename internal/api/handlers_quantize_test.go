package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openinfer/openinfer-studio/internal/convert"
	"github.com/openinfer/openinfer-studio/internal/quantize"
)

func decodeBody(t *testing.T, body string) (int, quantize.Request, bool) {
	t.Helper()
	r := httptest.NewRequest("POST", "/api/v1/quantize/jobs", strings.NewReader(body))
	w := httptest.NewRecorder()
	req, ok := decodeQuantizeRequest(w, r)
	return w.Code, req, ok
}

func TestDecodeQuantizeRequestEffort(t *testing.T) {
	code, req, ok := decodeBody(t, `{"kind":"adaptive_quantize","effort":"deep"}`)
	if !ok || code != 200 {
		t.Fatalf("valid effort rejected: code=%d", code)
	}
	if req.Effort != "deep" || req.AdaptiveMode != "" {
		t.Fatalf("effort fields = %#v, want effort=deep without adaptive_mode", req)
	}
}

func TestDecodeQuantizeRequestAbsentEffortLeavesBackendDefaults(t *testing.T) {
	_, req, ok := decodeBody(t, `{"kind":"adaptive_quantize"}`)
	if !ok {
		t.Fatal("body without effort rejected")
	}
	if req.Effort != "" || req.AdaptiveMode != "" || req.TargetBPW != 0 || req.TargetBytes != 0 {
		t.Fatalf("request should leave effort and target defaults to backend: %#v", req)
	}
}

func TestDecodeQuantizeRequestAdaptiveModeAlias(t *testing.T) {
	_, req, ok := decodeBody(t, `{"kind":"adaptive_quantize","adaptive_mode":"fast"}`)
	if !ok {
		t.Fatal("adaptive_mode alias rejected")
	}
	if req.Effort != "" || req.AdaptiveMode != "fast" {
		t.Fatalf("alias fields = %#v, want adaptive_mode=fast without effort", req)
	}
}

func TestDecodeQuantizeRequestEffortWinsOverAlias(t *testing.T) {
	_, req, ok := decodeBody(t, `{"effort":"deep","adaptive_mode":"fast"}`)
	if !ok {
		t.Fatal("effort + alias rejected")
	}
	if req.Effort != "deep" || req.AdaptiveMode != "" {
		t.Fatalf("effort should suppress alias forwarding: %#v", req)
	}
}

func TestDecodeQuantizeRequestExplicitEmptyEffortSuppressesAlias(t *testing.T) {
	_, req, ok := decodeBody(t, `{"effort":"","adaptive_mode":"fast"}`)
	if !ok {
		t.Fatal("explicit empty effort rejected")
	}
	if req.Effort != "" || req.AdaptiveMode != "" {
		t.Fatalf("explicit effort should suppress alias forwarding: %#v", req)
	}
}

func TestDecodeQuantizeRequestInvalidEffort(t *testing.T) {
	code, _, ok := decodeBody(t, `{"effort":"ludicrous"}`)
	if ok || code != 400 {
		t.Fatalf("invalid effort accepted: code=%d ok=%v", code, ok)
	}
}

func TestDecodeQuantizeRequestInvalidAlias(t *testing.T) {
	code, _, ok := decodeBody(t, `{"adaptive_mode":"ludicrous"}`)
	if ok || code != 400 {
		t.Fatalf("invalid adaptive_mode accepted: code=%d ok=%v", code, ok)
	}
}

func TestDecodeQuantizeRequestPresetPassesThrough(t *testing.T) {
	r := httptest.NewRequest("POST", "/api/v1/quantize/jobs",
		strings.NewReader(`{"kind":"adaptive_quantize","adaptive_preset":"compact"}`))
	w := httptest.NewRecorder()
	req, ok := decodeQuantizeRequest(w, r)
	if !ok {
		t.Fatal("adaptive_preset rejected")
	}
	if req.AdaptivePreset != "compact" {
		t.Fatalf("preset not passed through: %q", req.AdaptivePreset)
	}
}

func TestDecodeFromHFPreviewRequest(t *testing.T) {
	t.Run("repo only remains generic", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/api/v1/quantize/from-hf/preview?repo=org/model", nil)
		w := httptest.NewRecorder()
		repo, req, dynamic, ok := decodeFromHFPreviewRequest(w, r)
		if !ok || dynamic || repo != "org/model" {
			t.Fatalf("decoded = repo=%q dynamic=%v ok=%v", repo, dynamic, ok)
		}
		if req.Effort != "" || req.TargetBPW != 0 || req.TargetBytes != 0 {
			t.Fatalf("generic request must not add Dynamic sizing: %#v", req)
		}
	})

	t.Run("dynamic carries request options", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/api/v1/quantize/from-hf/preview?repo=org/model&dynamic=true&effort=deep&target_bpw=3.83&target_bytes=123&generate_imatrix=false", nil)
		w := httptest.NewRecorder()
		_, req, dynamic, ok := decodeFromHFPreviewRequest(w, r)
		if !ok || !dynamic {
			t.Fatalf("Dynamic request rejected: code=%d", w.Code)
		}
		if req.Kind != quantize.KindFromHF || req.Effort != "deep" || req.TargetBPW != 3.83 || req.TargetBytes != 123 || req.GenerateIMatrix {
			t.Fatalf("request = %#v", req)
		}
	})

	t.Run("dynamic defaults backend effort and target", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/api/v1/quantize/from-hf/preview?repo=org/model&dynamic=true", nil)
		w := httptest.NewRecorder()
		_, req, dynamic, ok := decodeFromHFPreviewRequest(w, r)
		if !ok || !dynamic || req.Effort != "profiled" || req.TargetBPW != 0 || !req.GenerateIMatrix {
			t.Fatalf("default Dynamic request = %#v, dynamic=%v ok=%v", req, dynamic, ok)
		}
	})

	t.Run("rejects invalid effort", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/api/v1/quantize/from-hf/preview?repo=org/model&effort=ludicrous", nil)
		w := httptest.NewRecorder()
		_, _, _, ok := decodeFromHFPreviewRequest(w, r)
		if ok || w.Code != http.StatusBadRequest {
			t.Fatalf("invalid effort accepted: code=%d ok=%v", w.Code, ok)
		}
	})
}

func TestFromHFPreviewDynamicSizing(t *testing.T) {
	q := quantize.NewManager(nil, nil, nil, nil, nil, nil, nil, nil)
	q.SetProbeFn(func(context.Context, string) (*quantize.FromHFPreview, error) {
		return &quantize.FromHFPreview{ProbeResult: convert.ProbeResult{
			Compatible: true, SnapshotBytes: 100, EstimatedGGUFBytes: 1000,
		}, DiskPeakBytes: 777}, nil
	})
	h := &handlers{d: &Deps{Quant: q}}

	preview := func(query string) quantize.FromHFPreview {
		t.Helper()
		r := httptest.NewRequest(http.MethodGet, "/api/v1/quantize/from-hf/preview?repo=org/model"+query, nil)
		w := httptest.NewRecorder()
		h.quantizeFromHFPreview(w, r)
		if w.Code != http.StatusOK {
			t.Fatalf("preview failed: code=%d body=%s", w.Code, w.Body.String())
		}
		var out quantize.FromHFPreview
		if err := json.NewDecoder(w.Body).Decode(&out); err != nil {
			t.Fatal(err)
		}
		return out
	}

	if generic := preview(""); generic.DiskPeakBytes != 777 {
		t.Fatalf("generic preview used Dynamic sizing: disk_peak_bytes=%d", generic.DiskPeakBytes)
	}
	fast := preview("&dynamic=true&effort=fast&target_bpw=3.83&generate_imatrix=true")
	profiled := preview("&dynamic=true&effort=profiled&target_bpw=3.83&generate_imatrix=true")
	deep := preview("&dynamic=true&effort=deep&target_bpw=3.83&generate_imatrix=true")
	if !(fast.DiskPeakBytes < profiled.DiskPeakBytes && profiled.DiskPeakBytes < deep.DiskPeakBytes) {
		t.Fatalf("Dynamic preview peaks = fast %d, profiled %d, deep %d", fast.DiskPeakBytes, profiled.DiskPeakBytes, deep.DiskPeakBytes)
	}
}

func TestDecodeQuantizeRequestQuantTier(t *testing.T) {
	for _, tier := range []string{"q5", "q4", "q3", "q2"} {
		code, req, ok := decodeBody(t, `{"kind":"adaptive_quantize","quant_tier":"`+tier+`"}`)
		if !ok || code != 200 {
			t.Fatalf("valid tier %q rejected: code=%d", tier, code)
		}
		if req.QuantTier != tier {
			t.Fatalf("tier %q not forwarded: %#v", tier, req)
		}
	}

	t.Run("case-insensitive and trimmed", func(t *testing.T) {
		_, req, ok := decodeBody(t, `{"kind":"adaptive_quantize","quant_tier":"  Q4  "}`)
		if !ok {
			t.Fatal("Q4 with spaces rejected")
		}
		if req.QuantTier != "q4" {
			t.Fatalf("tier not normalized: %q", req.QuantTier)
		}
	})

	t.Run("invalid rejected", func(t *testing.T) {
		code, _, ok := decodeBody(t, `{"kind":"adaptive_quantize","quant_tier":"q9"}`)
		if ok || code != 400 {
			t.Fatalf("invalid tier accepted: code=%d ok=%v", code, ok)
		}
	})

	t.Run("custom with target_bytes accepted", func(t *testing.T) {
		code, req, ok := decodeBody(t, `{"kind":"adaptive_quantize","quant_tier":"custom","target_bytes":123}`)
		if !ok || code != 200 {
			t.Fatalf("custom with bytes rejected: code=%d", code)
		}
		if req.QuantTier != "custom" || req.TargetBytes != 123 {
			t.Fatalf("custom request = %#v", req)
		}
	})

	t.Run("custom without target_bytes rejected", func(t *testing.T) {
		code, _, ok := decodeBody(t, `{"kind":"adaptive_quantize","quant_tier":"custom"}`)
		if ok || code != 400 {
			t.Fatalf("custom without bytes accepted: code=%d ok=%v", code, ok)
		}
	})

	t.Run("empty defaults to q4 at the backend", func(t *testing.T) {
		code, req, ok := decodeBody(t, `{"kind":"adaptive_quantize"}`)
		if !ok || code != 200 {
			t.Fatalf("empty tier rejected: code=%d", code)
		}
		if req.QuantTier != "" {
			t.Fatalf("empty tier should pass through unset: %#v", req)
		}
		// The backend resolves an empty tier to the q4 default (4.5 BPW);
		// see quantize.resolveQuantTier / normalizeAdaptiveTarget.
		if bpw, ok := quantize.TierBPW("q4"); !ok || bpw != 4.5 {
			t.Fatalf("q4 default BPW = %v ok=%v", bpw, ok)
		}
	})
}

func TestDecodeFromHFPreviewRequestQuantTier(t *testing.T) {
	t.Run("dynamic carries quant_tier", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/api/v1/quantize/from-hf/preview?repo=org/model&dynamic=true&quant_tier=q3", nil)
		w := httptest.NewRecorder()
		_, req, dynamic, ok := decodeFromHFPreviewRequest(w, r)
		if !ok || !dynamic || req.QuantTier != "q3" {
			t.Fatalf("request = %#v dynamic=%v ok=%v", req, dynamic, ok)
		}
	})

	t.Run("custom requires target_bytes", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/api/v1/quantize/from-hf/preview?repo=org/model&dynamic=true&quant_tier=custom", nil)
		w := httptest.NewRecorder()
		_, _, _, ok := decodeFromHFPreviewRequest(w, r)
		if ok || w.Code != http.StatusBadRequest {
			t.Fatalf("custom without bytes accepted: code=%d ok=%v", w.Code, ok)
		}
	})

	t.Run("invalid tier rejected", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/api/v1/quantize/from-hf/preview?repo=org/model&dynamic=true&quant_tier=big", nil)
		w := httptest.NewRecorder()
		_, _, _, ok := decodeFromHFPreviewRequest(w, r)
		if ok || w.Code != http.StatusBadRequest {
			t.Fatalf("invalid tier accepted: code=%d ok=%v", w.Code, ok)
		}
	})
}
