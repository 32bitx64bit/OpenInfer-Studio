package workflow

import (
	"context"
	"errors"
	"image"
	"image/color"
	"os"
	"path/filepath"
	"testing"
)

func TestImageOperationsPreserveGeometryAndPixels(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 3, 2))
	src.SetNRGBA(0, 0, color.NRGBA{R: 255, A: 255})
	src.SetNRGBA(2, 1, color.NRGBA{B: 255, A: 255})
	rotate, err := transformImage(src, &ImageStage{Op: "image.rotate", Width: 2, Height: 3, Params: map[string]any{"angle": "90"}})
	if err != nil || rotate.NRGBAAt(1, 0).R != 255 || rotate.NRGBAAt(0, 2).B != 255 {
		t.Fatalf("rotation pixels failed: %v", err)
	}
	crop, err := transformImage(src, &ImageStage{Op: "image.crop", Width: 1, Height: 1, Params: map[string]any{"x": 2, "y": 1}})
	if err != nil || crop.NRGBAAt(0, 0).B != 255 {
		t.Fatal("crop chose wrong region")
	}
	pad, err := transformImage(src, &ImageStage{Op: "image.pad", Width: 5, Height: 4, Params: map[string]any{"left": 1, "top": 1}})
	if err != nil || pad.NRGBAAt(1, 1).R != 255 || pad.NRGBAAt(0, 0).A != 0 {
		t.Fatal("padding must retain image position and transparent border")
	}
}

func TestImageOperationsResolveNativeOutputDimensions(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 3, 2))
	src.SetNRGBA(2, 1, color.NRGBA{R: 255, A: 255})
	rotated, err := transformImage(src, &ImageStage{Op: "image.rotate", Params: map[string]any{"angle": "90"}})
	if err != nil || rotated.Bounds().Dx() != 2 || rotated.Bounds().Dy() != 3 || rotated.NRGBAAt(0, 2).R != 255 {
		t.Fatalf("runtime rotate dimensions: %v", err)
	}
	padded, err := transformImage(src, &ImageStage{Op: "image.pad", Params: map[string]any{"left": 2, "bottom": 1}})
	if err != nil || padded.Bounds().Dx() != 5 || padded.Bounds().Dy() != 3 || padded.NRGBAAt(4, 1).R != 255 {
		t.Fatalf("runtime pad dimensions: %v", err)
	}
	if _, err := transformImage(src, &ImageStage{Op: "image.pad", Params: map[string]any{"left": 4096}}); err == nil {
		t.Fatal("unknown-size source bypassed output limits")
	}
}

func TestMaskBrushPaintAndErase(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source.png")
	if err := writeImage(root, "source.png", image.NewGray(image.Rect(0, 0, 32, 16))); err != nil {
		t.Fatal(err)
	}
	path, err := PaintMask(root, source, []MaskStroke{{Points: [][2]float64{{0.2, 0.5}, {0.8, 0.5}}, Radius: 0.1}, {Points: [][2]float64{{0.5, 0.5}}, Radius: 0.05, Erase: true}})
	if err != nil {
		t.Fatal(err)
	}
	img, err := readImage(path)
	if err != nil {
		t.Fatal(err)
	}
	if c := color.GrayModel.Convert(img.At(8, 8)).(color.Gray); c.Y != 255 {
		t.Fatal("brush stroke should paint white")
	}
	if c := color.GrayModel.Convert(img.At(15, 8)).(color.Gray); c.Y != 0 {
		t.Fatal("erase should restore black")
	}
	if _, err := PaintMask(root, source, []MaskStroke{{Radius: 0.1, Points: [][2]float64{{2, 0}}}}); err == nil {
		t.Fatal("out-of-image coordinates accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := PaintMaskContext(ctx, root, source, nil); !errors.Is(err, context.Canceled) {
		t.Fatal("canceled request wrote a mask")
	}
}

func TestWorkflowFileRoundTripPreservesEditorFields(t *testing.T) {
	store := newStore(t)
	g := txt2img()
	g.Nodes[0].Collapsed = true
	g.Groups = []Group{{Title: "Inputs", Nodes: []string{"n1"}, Note: "shared settings", Color: "#ff0000"}}
	rec, err := store.Create("Portable graph", graphJSON(t, g))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "workflow.json")
	if err := store.Export(rec.ID, path); err != nil {
		t.Fatal(err)
	}
	imported, err := store.Import(path)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ParseGraph(imported.Graph)
	if err != nil || !got.Nodes[0].Collapsed || got.Groups[0].Note != "shared settings" {
		t.Fatalf("lost editor fields: %+v, %v", got, err)
	}
	if err := os.WriteFile(path, []byte(`{"version":99}`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Import(path); err == nil {
		t.Fatal("unsupported file was imported")
	}
}
