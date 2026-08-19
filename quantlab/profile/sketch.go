package profile

import (
	"math"
	"math/rand"
)

// MakeSketches builds P×nIn Gaussian rows ~ N(0, diag(σ)) with σ from
// per-channel importance (sqrt of activation power). Deterministic for seed.
func MakeSketches(nIn, p int, sigma []float32, seed int64) [][]float32 {
	if nIn <= 0 {
		return nil
	}
	if p <= 0 {
		p = 32
	}
	rng := rand.New(rand.NewSource(seed + int64(nIn)*17))
	out := make([][]float32, p)
	for i := 0; i < p; i++ {
		row := make([]float32, nIn)
		for c := 0; c < nIn; c++ {
			s := 1.0
			if c < len(sigma) && sigma[c] > 0 {
				s = math.Sqrt(float64(sigma[c]))
			}
			row[c] = float32(rng.NormFloat64() * s)
		}
		out[i] = row
	}
	return out
}
