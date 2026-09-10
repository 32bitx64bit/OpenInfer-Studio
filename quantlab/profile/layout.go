package profile

// ImportanceLayout maps a retained imatrix vector onto a weight tensor.
//
// llama-imatrix accumulates activation power per INPUT channel of each
// matmul: for a weight of GGUF shape [ne0, rows] (ne0 = input features,
// rows = output features) the stored in_sum2 vector has length ne0, and every
// row of the weight is weighted by the same ne0 values. Fused 3-D expert
// stacks store one such vector per expert, concatenated in expert order; the
// stack's rows are laid out in matching per-expert blocks.
//
// Element (row r, column c) is therefore weighted by
// Values[expert(r)*NE0 + c] with expert(r) = r / (Rows/Experts).
type ImportanceLayout struct {
	NE0     uint64
	Rows    uint64
	Experts uint64
}

// LayoutFor validates values against a [ne0 x rows] tensor. It returns
// ok=false when the vector is empty or is not a whole number of ne0-length
// channel vectors that partitions rows evenly; callers then fall back to
// uniform weights.
func LayoutFor(values []float32, ne0, rows uint64) (ImportanceLayout, bool) {
	n := uint64(len(values))
	if n == 0 || ne0 == 0 || rows == 0 || n%ne0 != 0 {
		return ImportanceLayout{}, false
	}
	experts := n / ne0
	if experts == 0 || rows%experts != 0 {
		return ImportanceLayout{}, false
	}
	return ImportanceLayout{NE0: ne0, Rows: rows, Experts: experts}, true
}

// Row returns the ne0-length per-channel importance slice governing row r.
// The slice aliases values.
func (l ImportanceLayout) Row(values []float32, r uint64) []float32 {
	e := uint64(0)
	if l.Experts > 1 {
		e = r / (l.Rows / l.Experts)
		if e >= l.Experts {
			e = l.Experts - 1
		}
	}
	return values[e*l.NE0 : (e+1)*l.NE0]
}

// Fill expands the importance of rows [rowStart, rowStart+n) into dst
// (length n*ne0, row-major) as per-element weights.
func (l ImportanceLayout) Fill(dst, values []float32, rowStart, n uint64) {
	for i := uint64(0); i < n; i++ {
		copy(dst[i*l.NE0:(i+1)*l.NE0], l.Row(values, rowStart+i))
	}
}

// ChannelMean returns the per-input-channel importance averaged over
// experts (identity for 2-D weights), or nil when values do not fit the
// tensor. Non-finite or negative entries count as zero.
func ChannelMean(values []float32, ne0, rows uint64) []float64 {
	l, ok := LayoutFor(values, ne0, rows)
	if !ok {
		return nil
	}
	out := make([]float64, ne0)
	for e := uint64(0); e < l.Experts; e++ {
		v := values[e*ne0 : (e+1)*ne0]
		for c := range out {
			x := float64(v[c])
			if x > 0 && x == x && x < 1e300 {
				out[c] += x
			}
		}
	}
	if l.Experts > 1 {
		for c := range out {
			out[c] /= float64(l.Experts)
		}
	}
	return out
}
