package workflow

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"

	"github.com/google/uuid"
)

type MaskStroke struct {
	Points [][2]float64 `json:"points"`
	Radius float64      `json:"radius"`
	Erase  bool         `json:"erase"`
}

// PaintMask maps normalized image coordinates to a grayscale inpainting mask.
func PaintMask(root, path string, strokes []MaskStroke) (string, error) {
	return PaintMaskContext(context.Background(), root, path, strokes)
}

func PaintMaskContext(ctx context.Context, root, path string, strokes []MaskStroke) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if !filepath.IsAbs(path) {
		return "", invalid("input image path must be absolute")
	}
	f, err := os.Open(path)
	if err != nil {
		return "", invalid("input image: %v", err)
	}
	cfg, _, err := image.DecodeConfig(f)
	f.Close()
	if err != nil {
		return "", invalid("mask painting needs PNG, JPEG or GIF")
	}
	w, h := cfg.Width, cfg.Height
	if w < 1 || h < 1 || w > 4096 || h > 4096 {
		return "", invalid("mask dimensions must be at most 4096 × 4096")
	}
	if len(strokes) > 1000 {
		return "", invalid("too many mask strokes")
	}
	points := 0
	for _, s := range strokes {
		points += len(s.Points)
		if s.Radius <= 0 || s.Radius > 0.5 || math.IsNaN(s.Radius) || math.IsInf(s.Radius, 0) {
			return "", invalid("invalid brush radius")
		}
		for _, p := range s.Points {
			for _, v := range p {
				if v < 0 || v > 1 || math.IsNaN(v) || math.IsInf(v, 0) {
					return "", invalid("brush point must be inside the image")
				}
			}
		}
	}
	if points > 20000 {
		return "", invalid("mask has too many points")
	}
	out := image.NewGray(image.Rect(0, 0, w, h))
	var work int64
	for _, s := range strokes {
		radius := s.Radius * float64(max(w, h))
		value := uint8(255)
		if s.Erase {
			value = 0
		}
		for i, p := range s.Points {
			ax, ay := p[0]*float64(w-1), p[1]*float64(h-1)
			bx, by := ax, ay
			if i > 0 {
				bx, by = s.Points[i-1][0]*float64(w-1), s.Points[i-1][1]*float64(h-1)
			}
			x0, x1 := max(0, int(math.Floor(min(ax, bx)-radius))), min(w-1, int(math.Ceil(max(ax, bx)+radius)))
			y0, y1 := max(0, int(math.Floor(min(ay, by)-radius))), min(h-1, int(math.Ceil(max(ay, by)+radius)))
			work += int64(x1-x0+1) * int64(y1-y0+1)
			if work > 300_000_000 {
				return "", invalid("mask is too complex; reduce brush size or simplify strokes")
			}
			dx, dy := bx-ax, by-ay
			length := dx*dx + dy*dy
			for y := y0; y <= y1; y++ {
				if err := ctx.Err(); err != nil {
					return "", err
				}
				for x := x0; x <= x1; x++ {
					t := 0.0
					if length > 0 {
						t = math.Max(0, math.Min(1, ((float64(x)-ax)*dx+(float64(y)-ay)*dy)/length))
					}
					ex, ey := float64(x)-(ax+t*dx), float64(y)-(ay+t*dy)
					if ex*ex+ey*ey <= radius*radius {
						out.SetGray(x, y, color.Gray{Y: value})
					}
				}
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	rel := filepath.ToSlash(filepath.Join("images", "mask-"+uuid.NewString()+".png"))
	if err := writeImage(root, rel, out); err != nil {
		return "", fmt.Errorf("save mask: %w", err)
	}
	return filepath.Join(root, filepath.FromSlash(rel)), nil
}
