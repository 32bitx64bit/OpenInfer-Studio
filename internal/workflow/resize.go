package workflow

import (
	"image"
	"image/color"
	"math"
)

// resizeImage scales src to w x h. filter is "lanczos" (Lanczos-3, sharper,
// the default for hires passes) or "bilinear". It works on a straight-alpha
// copy and never allocates more than the destination plus one row pass.
func resizeImage(src image.Image, w, h int, filter string) *image.NRGBA {
	b := src.Bounds()
	sw, sh := b.Dx(), b.Dy()
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))
	if sw == 0 || sh == 0 || w == 0 || h == 0 {
		return dst
	}
	in := image.NewNRGBA(image.Rect(0, 0, sw, sh))
	for y := 0; y < sh; y++ {
		for x := 0; x < sw; x++ {
			in.SetNRGBA(x, y, color.NRGBAModel.Convert(src.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA))
		}
	}
	kernel, support := lanczos3, 3.0
	if filter == "bilinear" {
		kernel, support = triangle, 1.0
	}
	// Horizontal pass into tmp (w x sh), then vertical pass into dst.
	hx := buildWeights(sw, w, kernel, support)
	tmp := image.NewNRGBA(image.Rect(0, 0, w, sh))
	for y := 0; y < sh; y++ {
		for x := 0; x < w; x++ {
			var r, g, bl, a float64
			for _, wt := range hx[x] {
				c := in.NRGBAAt(wt.index, y)
				al := float64(c.A) * wt.weight
				r += float64(c.R) * al
				g += float64(c.G) * al
				bl += float64(c.B) * al
				a += al
			}
			tmp.SetNRGBA(x, y, premulToNRGBA(r, g, bl, a))
		}
	}
	vy := buildWeights(sh, h, kernel, support)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var r, g, bl, a float64
			for _, wt := range vy[y] {
				c := tmp.NRGBAAt(x, wt.index)
				al := float64(c.A) * wt.weight
				r += float64(c.R) * al
				g += float64(c.G) * al
				bl += float64(c.B) * al
				a += al
			}
			dst.SetNRGBA(x, y, premulToNRGBA(r, g, bl, a))
		}
	}
	return dst
}

type tap struct {
	index  int
	weight float64
}

// buildWeights returns, for each destination pixel, the source taps and
// normalised weights. When shrinking, the kernel widens by the scale factor
// so detail is averaged rather than skipped.
func buildWeights(srcN, dstN int, kernel func(float64) float64, support float64) [][]tap {
	scale := float64(srcN) / float64(dstN)
	filterScale := math.Max(scale, 1)
	radius := support * filterScale
	out := make([][]tap, dstN)
	for i := 0; i < dstN; i++ {
		center := (float64(i) + 0.5) * scale
		lo := int(math.Floor(center - radius))
		hi := int(math.Ceil(center + radius))
		var taps []tap
		var sum float64
		for j := lo; j < hi; j++ {
			wgt := kernel((float64(j) + 0.5 - center) / filterScale)
			if wgt == 0 {
				continue
			}
			idx := j
			if idx < 0 {
				idx = 0
			}
			if idx >= srcN {
				idx = srcN - 1
			}
			taps = append(taps, tap{idx, wgt})
			sum += wgt
		}
		if sum != 0 {
			for k := range taps {
				taps[k].weight /= sum
			}
		}
		out[i] = taps
	}
	return out
}

func lanczos3(x float64) float64 {
	x = math.Abs(x)
	if x >= 3 {
		return 0
	}
	if x < 1e-9 {
		return 1
	}
	px := math.Pi * x
	return 3 * math.Sin(px) * math.Sin(px/3) / (px * px)
}

func triangle(x float64) float64 {
	x = math.Abs(x)
	if x >= 1 {
		return 0
	}
	return 1 - x
}

// premulToNRGBA converts alpha-weighted channel sums back to straight alpha.
func premulToNRGBA(r, g, b, a float64) color.NRGBA {
	if a <= 0 {
		return color.NRGBA{}
	}
	clamp := func(v float64) uint8 {
		v = math.Round(v)
		if v < 0 {
			return 0
		}
		if v > 255 {
			return 255
		}
		return uint8(v)
	}
	return color.NRGBA{R: clamp(r / a), G: clamp(g / a), B: clamp(b / a), A: clamp(a)}
}
