package workflow

import (
	"image"
	"image/color"
	"testing"
)

func solid(w, h int, c color.NRGBA) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
	return img
}

func TestResizeKeepsSolidColourAndSize(t *testing.T) {
	for _, filter := range []string{"lanczos", "bilinear"} {
		src := solid(40, 30, color.NRGBA{R: 200, G: 100, B: 50, A: 255})
		for _, dim := range [][2]int{{80, 60}, {20, 15}, {61, 47}} {
			out := resizeImage(src, dim[0], dim[1], filter)
			if out.Bounds().Dx() != dim[0] || out.Bounds().Dy() != dim[1] {
				t.Fatalf("%s: size = %v, want %v", filter, out.Bounds().Size(), dim)
			}
			for _, p := range [][2]int{{0, 0}, {dim[0] - 1, dim[1] - 1}, {dim[0] / 2, dim[1] / 2}} {
				got := out.NRGBAAt(p[0], p[1])
				if got != (color.NRGBA{R: 200, G: 100, B: 50, A: 255}) {
					t.Fatalf("%s %v: pixel %v = %+v, want the solid colour", filter, dim, p, got)
				}
			}
		}
	}
}

func TestResizeKeepsEdgesBetweenHalves(t *testing.T) {
	// Left half black, right half white: after doubling, the far left stays
	// black and the far right white, with a transition near the middle.
	src := image.NewNRGBA(image.Rect(0, 0, 10, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 10; x++ {
			v := uint8(0)
			if x >= 5 {
				v = 255
			}
			src.SetNRGBA(x, y, color.NRGBA{v, v, v, 255})
		}
	}
	out := resizeImage(src, 20, 8, "lanczos")
	if out.NRGBAAt(1, 3).R > 10 || out.NRGBAAt(18, 3).R < 245 {
		t.Fatalf("far pixels drifted: left %d right %d", out.NRGBAAt(1, 3).R, out.NRGBAAt(18, 3).R)
	}
	// The edge sits between destination pixels 9 and 10.
	if lo, hi := out.NRGBAAt(9, 3).R, out.NRGBAAt(10, 3).R; lo > 100 || hi < 155 {
		t.Fatalf("edge pixels = %d, %d; want dark then light", lo, hi)
	}
}

func TestResizeTransparentPixelsStayTransparent(t *testing.T) {
	src := solid(8, 8, color.NRGBA{})
	out := resizeImage(src, 16, 16, "lanczos")
	if out.NRGBAAt(5, 5).A != 0 {
		t.Fatal("transparent input must stay transparent")
	}
}
