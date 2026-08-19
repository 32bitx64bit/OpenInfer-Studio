package profile

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"

	"quantlab/core"
)

// CalibrationVersion is the on-disk schema version of a calibration store.
const CalibrationVersion = 1

// Calibration hyper-parameters.
const (
	// calMinSamples is the smallest sample count a level may be fitted with.
	calMinSamples = 12
	// calRidge is the L2 penalty on standardized features.
	calRidge = 1.0
	// calCorrMin/Max clamp the multiplicative correction applied to
	// heuristic losses so a sparse fit cannot reorder the landscape wildly.
	calCorrMin = 0.25
	calCorrMax = 4.0
)

// Calibration is one fitted multiplicative correction of heuristic
// per-weight losses toward measured behavior, in log10 space.
type Calibration struct {
	Features  int       `json:"features"`
	Beta      []float64 `json:"beta"`
	Mean      []float64 `json:"mean"`
	Std       []float64 `json:"std"`
	Intercept float64   `json:"intercept"`
	Samples   int       `json:"samples"`
	R2        float64   `json:"r2,omitempty"`
	FeatureAt string    `json:"featureAt,omitempty"`
	MinCorr   float64   `json:"minCorr"`
	MaxCorr   float64   `json:"maxCorr"`
}

// CalibrationStore persists fitted calibrations with a hierarchical
// fallback: model-specific first, then architecture family, then global.
type CalibrationStore struct {
	Version  int                     `json:"version"`
	ModelID  string                  `json:"modelID"`
	ModelSHA string                  `json:"modelSHA"`
	Arch     string                  `json:"arch,omitempty"`
	Levels   map[string]*Calibration `json:"levels"`
	Samples  map[string]int          `json:"samples"`
}

// Level keys.
const (
	LevelModel  = "model"
	LevelGlobal = "global"
)

// LevelArch builds the architecture-family level key.
func LevelArch(arch string) string { return "arch:" + arch }

// NewCalibrationStore returns an empty store bound to a model identity.
func NewCalibrationStore(modelID, modelSHA, arch string) *CalibrationStore {
	return &CalibrationStore{
		Version:  CalibrationVersion,
		ModelID:  modelID,
		ModelSHA: modelSHA,
		Arch:     arch,
		Levels:   map[string]*Calibration{},
		Samples:  map[string]int{},
	}
}

// Resolve picks the most specific fitted level: model, then arch, then
// global. Nil when nothing is fitted.
func (s *CalibrationStore) Resolve(arch string) *Calibration {
	if s == nil {
		return nil
	}
	if c := s.Levels[LevelModel]; c != nil && c.Samples >= calMinSamples {
		return c
	}
	if arch != "" {
		if c := s.Levels[LevelArch(arch)]; c != nil && c.Samples >= calMinSamples {
			return c
		}
	}
	if c := s.Levels[LevelGlobal]; c != nil && c.Samples >= calMinSamples {
		return c
	}
	return nil
}

// calSample is one (features, target) training pair.
type calSample struct {
	f    []float64
	y    float64
	w    float64
	name string
	d    core.DType
}

// CalFeatures extracts the deterministic feature vector describing one
// (tensor, dtype) heuristic estimate. The same function runs at fit and
// apply time; it must never change for a persisted calibration (bump
// CalibrationVersion and refit if it does).
func CalFeatures(t core.TensorDesc, d core.DType, est *FallbackEstimator) []float64 {
	bpw, _ := d.BitsPerWeight()
	heu := 1.0
	if est != nil {
		if l, _ := est.heuristic(t, d); l > 0 {
			heu = l / float64(t.Elements)
		}
	}
	st, hasStats := est.stats(t.Name)
	entropy, spik, impRel := -1.0, 1.0, 1.0
	if hasStats {
		entropy = st.Entropy
		spik = st.Spikiness
		if est.hasImatrix && est.imatrixMean > 0 {
			impRel = st.Mean / est.imatrixMean
		}
	}
	depth := 0.5
	if l := layerIndex(t.Name); l >= 0 && est != nil && est.maxLayer > 0 {
		depth = float64(l) / float64(est.maxLayer)
	}
	role := roleBase(t.Name)
	f := []float64{
		math.Log10(math.Max(heu, 1e-12)),
		bpw,
		boolF(d.RequiresImatrix()),
		boolF(isKQuant(d)),
		math.Log10(math.Max(float64(t.Elements), 1)),
		depth,
		entropy,
		math.Log10(math.Max(spik, 1e-6)),
		math.Log10(math.Max(impRel, 1e-6)),
		role,
	}
	return f
}

func boolF(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func isKQuant(d core.DType) bool {
	switch d.BaseTensorType() {
	case core.DTypeQ2_K, core.DTypeQ3_K, core.DTypeQ4_K_T, core.DTypeQ5_K_T, core.DTypeQ6_K, core.DTypeQ8_K:
		return true
	}
	return false
}

// FitCalibration fits the model-level calibration from measured cache
// entries. Samples are (features from CalFeatures, target = measured
// per-weight loss, weight = entry confidence). It returns nil when too few
// samples exist. Deterministic: cache entries are walked in sorted order.
func FitCalibration(bank *core.TensorBank, cache *Cache, est *FallbackEstimator, featureAt string) *Calibration {
	if cache == nil || bank == nil {
		return nil
	}
	keys := make([]string, 0, len(cache.Entries))
	byKey := map[string]CacheEntry{}
	for _, e := range cache.Entries {
		k := e.TensorName + "\x00" + string(e.Target)
		if _, dup := byKey[k]; !dup {
			keys = append(keys, k)
		}
		byKey[k] = e
	}
	sort.Strings(keys)
	var samples []calSample
	for _, k := range keys {
		e := byKey[k]
		t, ok := bank.Find(e.TensorName)
		if !ok {
			continue
		}
		if !(e.Loss > 0) {
			continue
		}
		samples = append(samples, calSample{
			f:    CalFeatures(t, e.Target, est),
			y:    math.Log10(e.Loss),
			w:    entryConfidence(e),
			name: e.TensorName,
			d:    e.Target,
		})
	}
	if len(samples) < calMinSamples {
		return nil
	}
	return fitRidge(samples, featureAt)
}

// fitRidge solves the standardized ridge regression via Gaussian
// elimination on the normal equations.
func fitRidge(samples []calSample, featureAt string) *Calibration {
	n := len(samples[0].f)
	for _, s := range samples {
		if len(s.f) != n {
			return nil
		}
	}
	mean := make([]float64, n)
	std := make([]float64, n)
	var wSum float64
	for _, s := range samples {
		wSum += s.w
	}
	if wSum <= 0 {
		wSum = 1
	}
	for j := 0; j < n; j++ {
		var m float64
		for _, s := range samples {
			m += s.w * s.f[j]
		}
		mean[j] = m / wSum
		var v float64
		for _, s := range samples {
			d := s.f[j] - mean[j]
			v += s.w * d * d
		}
		std[j] = math.Sqrt(v / wSum)
		if std[j] < 1e-9 {
			std[j] = 1
		}
	}
	X := make([][]float64, len(samples))
	y := make([]float64, len(samples))
	for i, s := range samples {
		row := make([]float64, n)
		for j := 0; j < n; j++ {
			row[j] = (s.f[j] - mean[j]) / std[j]
		}
		X[i] = row
		y[i] = s.y
	}
	// Normal equations (X^T W X + ridge I) beta = X^T W y.
	A := make([][]float64, n)
	for i := range A {
		A[i] = make([]float64, n+1)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for k, sm := range samples {
				s += sm.w * X[k][i] * X[k][j]
			}
			if i == j {
				s += calRidge
			}
			A[i][j] = s
		}
		var s float64
		for k, sm := range samples {
			s += sm.w * X[k][i] * y[k]
		}
		A[i][n] = s
	}
	if !gaussSolve(A) {
		return nil
	}
	beta := make([]float64, n)
	for i := range beta {
		beta[i] = A[i][n]
	}
	// Weighted intercept and R^2 in log space.
	var yBar, wS float64
	for k, sm := range samples {
		yBar += sm.w * y[k]
		wS += sm.w
	}
	if wS <= 0 {
		wS = 1
	}
	yBar /= wS
	var ssRes, ssTot float64
	for k, sm := range samples {
		p := 0.0
		for j := 0; j < n; j++ {
			p += beta[j] * X[k][j]
		}
		ssRes += sm.w * (y[k] - p) * (y[k] - p)
		ssTot += sm.w * (y[k] - yBar) * (y[k] - yBar)
	}
	r2 := 0.0
	if ssTot > 0 {
		r2 = 1 - ssRes/ssTot
	}
	return &Calibration{
		Features: n, Beta: beta, Mean: mean, Std: std,
		Intercept: yBar, Samples: len(samples), R2: r2, FeatureAt: featureAt,
		MinCorr: calCorrMin, MaxCorr: calCorrMax,
	}
}

// gaussSolve solves A x = b in place (augmented, n x n+1); reports success.
func gaussSolve(A [][]float64) bool {
	n := len(A)
	for col := 0; col < n; col++ {
		piv := col
		for r := col + 1; r < n; r++ {
			if math.Abs(A[r][col]) > math.Abs(A[piv][col]) {
				piv = r
			}
		}
		if math.Abs(A[piv][col]) < 1e-12 {
			return false
		}
		A[col], A[piv] = A[piv], A[col]
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := A[r][col] / A[col][col]
			for c := col; c <= n; c++ {
				A[r][c] -= f * A[col][c]
			}
		}
	}
	for i := range A {
		A[i][n] /= A[i][i]
	}
	return true
}

// PredictLog returns the calibrated log10 per-weight loss for one feature
// vector.
func (c *Calibration) PredictLog(f []float64) float64 {
	if c == nil || len(f) != c.Features {
		return 0
	}
	p := c.Intercept
	for j := range f {
		p += c.Beta[j] * ((f[j] - c.Mean[j]) / c.Std[j])
	}
	return p
}

// Correction returns the multiplicative correction of a heuristic per-weight
// loss heu for feature vector f, clamped to [MinCorr, MaxCorr].
func (c *Calibration) Correction(f []float64, heu float64) float64 {
	if c == nil || heu <= 0 || len(f) != c.Features {
		return 1
	}
	delta := c.PredictLog(f) - math.Log10(heu)
	corr := math.Pow(10, delta)
	return math.Max(c.MinCorr, math.Min(c.MaxCorr, corr))
}

// SaveCalibration writes the store deterministically.
func SaveCalibration(w io.Writer, s *CalibrationStore) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(s)
}

// LoadCalibration strictly decodes a store bound to the expected identity.
// A version mismatch is a hard error; an identity mismatch returns
// (nil, nil) so stale calibrations never leak into a run.
func LoadCalibration(r io.Reader, modelID, modelSHA string) (*CalibrationStore, error) {
	var s CalibrationStore
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&s); err != nil {
		return nil, fmt.Errorf("calibration: decode: %w", err)
	}
	if s.Version != CalibrationVersion {
		return nil, fmt.Errorf("calibration: unsupported version %d (want %d)", s.Version, CalibrationVersion)
	}
	if s.ModelID != modelID || (modelSHA != "" && s.ModelSHA != "" && s.ModelSHA != modelSHA) {
		return nil, nil
	}
	return &s, nil
}

// SetCalibration installs a resolved calibration on the estimator. The
// heuristic path multiplies its loss by the fitted correction; measured and
// exact paths are untouched. A nil calibration clears it.
func (e *FallbackEstimator) SetCalibration(c *Calibration) {
	if e == nil {
		return
	}
	e.calib = c
}

// calibrate applies the installed calibration to a heuristic loss.
func (e *FallbackEstimator) calibrate(t core.TensorDesc, target core.DType, loss float64) float64 {
	if e == nil || e.calib == nil || loss <= 0 {
		return loss
	}
	f := CalFeatures(t, target, e)
	return loss * e.calib.Correction(f, loss/float64(t.Elements))
}
