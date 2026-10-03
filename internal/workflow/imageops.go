package workflow

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"path/filepath"
)

type ImageStage struct {
	Src    ImageRef       `json:"src"`
	Other  *ImageRef      `json:"other,omitempty"`
	Op     string         `json:"op"`
	Params map[string]any `json:"params"`
	Width  int            `json:"width"`
	Height int            `json:"height"`
}

func imageSpecs() []NodeSpec {
	in := []Port{{Name: "image", Type: PortTypes{TypeImage}, Required: true}}
	out := []Port{{Name: "image", Type: PortTypes{TypeImage}}}
	integer := func(name string, def, min, max int) ParamSpec {
		return ParamSpec{Name: name, Kind: ParamInt, Default: def, Min: ptr(float64(min)), Max: ptr(float64(max))}
	}
	return []NodeSpec{
		{Type: "mask.load", Title: "Load Mask", Category: CatInput, Description: "White pixels are regenerated; black pixels are preserved. Connect to Sample's mask input.", Outputs: []Port{{Name: "mask", Type: PortTypes{TypeMask}}}, Params: []ParamSpec{{Name: "path", Kind: ParamPath, Required: true}}},
		{Type: "image.crop", Title: "Crop", Category: CatImage, Inputs: in, Outputs: out, Params: []ParamSpec{integer("x", 0, 0, 4096), integer("y", 0, 0, 4096), integer("width", 512, 1, 4096), integer("height", 512, 1, 4096)}},
		{Type: "image.pad", Title: "Pad Canvas", Category: CatImage, Description: "Extend the canvas with transparent pixels. Use a mask for outpainting.", Inputs: in, Outputs: out, Params: []ParamSpec{integer("left", 0, 0, 4096), integer("top", 0, 0, 4096), integer("right", 0, 0, 4096), integer("bottom", 0, 0, 4096)}},
		{Type: "image.rotate", Title: "Rotate / Flip", Category: CatImage, Inputs: in, Outputs: out, Params: []ParamSpec{{Name: "angle", Kind: ParamEnum, Options: []string{"0", "90", "180", "270"}, Default: "90"}, {Name: "flip_x", Kind: ParamBool, Default: false}, {Name: "flip_y", Kind: ParamBool, Default: false}}},
		{Type: "image.blend", Title: "Blend Images", Category: CatImage, Description: "Mix two images with identical dimensions.", Inputs: []Port{in[0], {Name: "other", Type: PortTypes{TypeImage}, Required: true}}, Outputs: out, Params: []ParamSpec{{Name: "opacity", Kind: ParamFloat, Default: 0.5, Min: ptr(0), Max: ptr(1)}}},
		{Type: "image.pick", Title: "Select Batch Image", Category: CatImage, Description: "Choose one image from a batch (index starts at zero).", Inputs: in, Outputs: out, Params: []ParamSpec{integer("index", 0, 0, 7)}},
		{Type: "note", Title: "Note", Category: CatPrompt, Description: "A note for organizing the workflow; it does not run.", Params: []ParamSpec{{Name: "text", Kind: ParamText, Default: ""}}},
	}
}

func (b *builder) imageOp(n Node) {
	up, _, ok := b.upstream(n, "image")
	if !ok {
		return
	}
	src, ok := b.imgs[up.ID]
	if !ok {
		return
	}
	p := b.effective(n)
	w, h := src.w, src.h
	var other *ImageRef
	switch n.Type {
	case "image.crop":
		w, h = p.integer("width"), p.integer("height")
		if src.w > 0 && src.h > 0 && (p.integer("x")+w > src.w || p.integer("y")+h > src.h) {
			b.errorf("plan.crop_size", n.ID, "", "crop exceeds the %d × %d input", src.w, src.h)
			return
		}
	case "image.pad":
		if w > 0 && h > 0 {
			w += p.integer("left") + p.integer("right")
			h += p.integer("top") + p.integer("bottom")
		}
	case "image.rotate":
		if p.str("angle") == "90" || p.str("angle") == "270" {
			w, h = h, w
		}
	case "image.blend":
		up, _, ok := b.upstream(n, "other")
		if !ok {
			return
		}
		o, ok := b.imgs[up.ID]
		if !ok {
			return
		}
		if o.w > 0 && w > 0 && (o.w != w || o.h != h) {
			b.errorf("plan.blend_size", n.ID, "other", "Blend requires identical dimensions; resize one input first")
			return
		}
		other = &o.ref
	case "image.pick":
		prev := b.stageByID(src.ref.Stage)
		count := 1
		if prev != nil && prev.Generate != nil {
			count = max(1, prev.Generate.Params.BatchCount)
		}
		if p.integer("index") >= count {
			b.errorf("plan.batch_index", n.ID, "", "index is outside the upstream batch")
			return
		}
	}
	if (w != 0 || h != 0) && (w < 1 || h < 1 || w > 4096 || h > 4096) {
		b.errorf("plan.image_size", n.ID, "", "result must be at most 4096 × 4096")
		return
	}
	st := b.newStage(n, StageImage)
	if n.Type != "image.pick" {
		b.warnBatch(n.ID, src.ref)
	}
	st.Image = &ImageStage{Src: src.ref, Other: other, Op: n.Type, Params: p, Width: w, Height: h}
	if src.ref.Stage != "" {
		st.Deps = append(st.Deps, src.ref.Stage)
	}
	if other != nil && other.Stage != "" && other.Stage != src.ref.Stage {
		st.Deps = append(st.Deps, other.Stage)
	}
	b.imgs[n.ID] = imgInfo{ref: ImageRef{Stage: st.ID}, w: w, h: h}
}

func readImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	cfg, _, err := image.DecodeConfig(f)
	if err != nil {
		return nil, fmt.Errorf("image operation needs PNG, JPEG or GIF: %w", err)
	}
	if cfg.Width <= 0 || cfg.Height <= 0 || int64(cfg.Width)*int64(cfg.Height) > maxResizePix {
		return nil, fmt.Errorf("image exceeds 64 megapixels")
	}
	if _, err := f.Seek(0, 0); err != nil {
		return nil, err
	}
	img, _, err := image.Decode(f)
	return img, err
}

func writeImage(root, rel string, img image.Image) error {
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".wf-image-*")
	if err != nil {
		return err
	}
	temp := f.Name()
	defer os.Remove(temp)
	if err := png.Encode(f, img); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(temp, path)
}

func (e *Executor) runImage(r *run, st Stage, prefix string) ([]string, error) {
	op := st.Image
	if op.Op == "image.pick" {
		ref := op.Src
		ref.Index = params(op.Params).integer("index")
		path, err := e.resolveImage(r, ref)
		if err != nil {
			return nil, err
		}
		rel := filepath.ToSlash(filepath.Join("images", fmt.Sprintf("wf-%s-%s.png", prefix, st.ID)))
		img, err := readImage(path)
		if err != nil {
			return nil, err
		}
		if err := writeImage(e.runner.MediaDir(), rel, img); err != nil {
			return nil, err
		}
		return []string{rel}, nil
	}
	path, err := e.resolveImage(r, op.Src)
	if err != nil {
		return nil, err
	}
	src, err := readImage(path)
	if err != nil {
		return nil, err
	}
	out, err := transformImage(src, op)
	if err != nil {
		return nil, err
	}
	if op.Other != nil {
		path, err := e.resolveImage(r, *op.Other)
		if err != nil {
			return nil, err
		}
		other, err := readImage(path)
		if err != nil {
			return nil, err
		}
		if other.Bounds().Dx() != out.Bounds().Dx() || other.Bounds().Dy() != out.Bounds().Dy() {
			return nil, fmt.Errorf("blend input dimensions changed")
		}
		alpha := params(op.Params).num("opacity")
		for y := 0; y < out.Bounds().Dy(); y++ {
			if r.ctx.Err() != nil {
				return nil, r.ctx.Err()
			}
			for x := 0; x < out.Bounds().Dx(); x++ {
				a := color.NRGBAModel.Convert(out.At(x, y)).(color.NRGBA)
				b := color.NRGBAModel.Convert(other.At(x+other.Bounds().Min.X, y+other.Bounds().Min.Y)).(color.NRGBA)
				mix := func(a, b uint8) uint8 { return uint8(float64(a)*(1-alpha) + float64(b)*alpha + 0.5) }
				out.SetNRGBA(x, y, color.NRGBA{R: mix(a.R, b.R), G: mix(a.G, b.G), B: mix(a.B, b.B), A: mix(a.A, b.A)})
			}
		}
	}
	if err := r.ctx.Err(); err != nil {
		return nil, err
	}
	rel := filepath.ToSlash(filepath.Join("images", fmt.Sprintf("wf-%s-%s.png", prefix, st.ID)))
	if err := writeImage(e.runner.MediaDir(), rel, out); err != nil {
		return nil, err
	}
	return []string{rel}, nil
}

func transformImage(src image.Image, op *ImageStage) (*image.NRGBA, error) {
	p := params(op.Params)
	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	resolved := *op
	if resolved.Width == 0 && resolved.Height == 0 {
		resolved.Width, resolved.Height = w, h
		switch op.Op {
		case "image.pad":
			resolved.Width += p.integer("left") + p.integer("right")
			resolved.Height += p.integer("top") + p.integer("bottom")
		case "image.rotate":
			if p.str("angle") == "90" || p.str("angle") == "270" {
				resolved.Width, resolved.Height = h, w
			}
		}
	}
	op = &resolved
	if op.Width < 1 || op.Height < 1 || op.Width > 4096 || op.Height > 4096 {
		return nil, fmt.Errorf("result must be at most 4096 × 4096")
	}
	out := image.NewNRGBA(image.Rect(0, 0, op.Width, op.Height))
	switch op.Op {
	case "image.crop":
		x, y := p.integer("x"), p.integer("y")
		if x < 0 || y < 0 || x+op.Width > w || y+op.Height > h {
			return nil, fmt.Errorf("crop input dimensions changed")
		}
		draw.Draw(out, out.Bounds(), src, bounds.Min.Add(image.Pt(x, y)), draw.Src)
	case "image.pad":
		draw.Draw(out, image.Rect(p.integer("left"), p.integer("top"), p.integer("left")+w, p.integer("top")+h), src, bounds.Min, draw.Src)
	case "image.rotate":
		angle := p.str("angle")
		expectedW, expectedH := w, h
		if angle == "90" || angle == "270" {
			expectedW, expectedH = h, w
		}
		if expectedW != op.Width || expectedH != op.Height {
			return nil, fmt.Errorf("rotate input dimensions changed")
		}
		flipX, _ := op.Params["flip_x"].(bool)
		flipY, _ := op.Params["flip_y"].(bool)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				sx, sy := x, y
				if flipX {
					sx = w - 1 - x
				}
				if flipY {
					sy = h - 1 - y
				}
				dx, dy := x, y
				switch angle {
				case "90":
					dx, dy = h-1-y, x
				case "180":
					dx, dy = w-1-x, h-1-y
				case "270":
					dx, dy = y, w-1-x
				}
				out.Set(dx, dy, src.At(bounds.Min.X+sx, bounds.Min.Y+sy))
			}
		}
	case "image.blend":
		if w != op.Width || h != op.Height {
			return nil, fmt.Errorf("blend input dimensions changed")
		}
		draw.Draw(out, out.Bounds(), src, bounds.Min, draw.Src)
	default:
		return nil, fmt.Errorf("unsupported image operation %s", op.Op)
	}
	return out, nil
}
