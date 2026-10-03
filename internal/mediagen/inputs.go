package mediagen

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

const (
	maxReferenceImages = 8
	maxInputBytes      = 64 << 20
	maxInputDimension  = 8192
)

type imageInput struct{ field, inline, path string }

func generationInputs(p GenerateParams) []imageInput {
	inputs := []imageInput{
		{"init_image", p.InitImage, p.InitImagePath}, {"mask_image", p.MaskImage, p.MaskImagePath},
		{"control_image", p.ControlImage, p.ControlImagePath},
	}
	for _, s := range p.RefImages {
		inputs = append(inputs, imageInput{field: "ref_images", inline: s})
	}
	for _, s := range p.RefImagePaths {
		inputs = append(inputs, imageInput{field: "ref_image_paths", path: s})
	}
	return inputs
}

func validateGenerationInputs(p GenerateParams) error {
	if len(p.Prompt) > 8000 || len(p.NegativePrompt) > 8000 {
		return fmt.Errorf("prompt exceeds 8000 byte limit")
	}
	for _, f := range []struct {
		name  string
		value float64
	}{{"cfg_scale", p.CFGScale}, {"guidance", p.Guidance}, {"strength", p.Strength}, {"control_strength", p.ControlStrength}} {
		if math.IsNaN(f.value) || math.IsInf(f.value, 0) {
			return fmt.Errorf("%s must be finite", f.name)
		}
	}
	if p.ControlStrength < 0 || p.ControlStrength > 10 {
		return fmt.Errorf("control_strength must be between 0 and 10")
	}
	if p.ControlStrength != 0 && p.ControlImage == "" && p.ControlImagePath == "" {
		return fmt.Errorf("control_strength requires control_image")
	}
	if (p.MaskImage != "" || p.MaskImagePath != "") && p.InitImage == "" && p.InitImagePath == "" {
		return fmt.Errorf("mask_image requires init_image")
	}
	if len(p.RefImages)+len(p.RefImagePaths) > maxReferenceImages {
		return fmt.Errorf("ref_images exceeds %d image limit", maxReferenceImages)
	}
	if len(p.RefImages) > 0 && len(p.RefImagePaths) > 0 {
		return fmt.Errorf("use either ref_images or ref_image_paths")
	}
	var inlineTotal int
	var total int
	for _, input := range generationInputs(p) {
		if input.inline == "" && input.path == "" {
			continue
		}
		if input.inline != "" && input.path != "" {
			return fmt.Errorf("use either %s or %s_path", input.field, input.field)
		}
		inlineTotal += len(input.inline)
		if inlineTotal > maxInitImageInlineBytes {
			return fmt.Errorf("inline images exceed %d MiB combined limit; use local image paths", maxInitImageInlineBytes>>20)
		}
		raw, _, err := readImageInput(input)
		if err != nil {
			return err
		}
		total += len(raw)
		if total > maxInputBytes {
			return fmt.Errorf("image inputs exceed %d MiB combined limit", maxInputBytes>>20)
		}
	}
	for _, s := range append(append([]string(nil), p.RefImages...), p.RefImagePaths...) {
		if s == "" {
			return fmt.Errorf("reference images must not be empty")
		}
	}
	if len(p.Lora) > 16 {
		return fmt.Errorf("lora exceeds 16 entry limit")
	}
	for _, l := range p.Lora {
		if err := validateLora(l); err != nil {
			return err
		}
	}
	return nil
}

func readImageInput(input imageInput) ([]byte, string, error) {
	var raw []byte
	var err error
	if input.path != "" {
		if err = validateLocalImagePath(input.path, input.field+"_path"); err != nil {
			return nil, "", err
		}
		f, err := os.Open(input.path)
		if err != nil {
			return nil, "", fmt.Errorf("%s_path: %w", input.field, err)
		}
		defer f.Close()
		st, err := f.Stat()
		if err != nil || !st.Mode().IsRegular() {
			return nil, "", fmt.Errorf("%s_path must be a regular file", input.field)
		}
		raw, err = io.ReadAll(io.LimitReader(f, maxInitImagePathBytes+1))
		if err != nil {
			return nil, "", fmt.Errorf("reading %s_path: %w", input.field, err)
		}
		if len(raw) > maxInitImagePathBytes {
			return nil, "", fmt.Errorf("%s_path exceeds %d MiB limit", input.field, maxInitImagePathBytes>>20)
		}
	} else {
		if len(input.inline) > maxInitImageInlineBytes {
			return nil, "", fmt.Errorf("%s exceeds %d MiB limit", input.field, maxInitImageInlineBytes>>20)
		}
		encoded := input.inline
		if strings.HasPrefix(encoded, "data:") {
			comma := strings.Index(encoded, ",")
			if comma < 0 || !strings.HasPrefix(encoded, "data:image/") || !strings.HasSuffix(encoded[:comma], ";base64") {
				return nil, "", fmt.Errorf("%s must be a base64 image data URL", input.field)
			}
			encoded = encoded[comma+1:]
		}
		raw, err = base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return nil, "", fmt.Errorf("%s must contain valid base64", input.field)
		}
	}
	w, h, mime, err := imageInfo(raw)
	if err != nil {
		return nil, "", fmt.Errorf("%s: %w", input.field, err)
	}
	if w < 1 || h < 1 || w > maxInputDimension || h > maxInputDimension {
		return nil, "", fmt.Errorf("%s dimensions must be 1..%d", input.field, maxInputDimension)
	}
	if input.path != "" && initImageExts[strings.ToLower(filepath.Ext(input.path))] != mime {
		return nil, "", fmt.Errorf("%s_path extension does not match image content", input.field)
	}
	return raw, mime, nil
}

// imageInfo reads geometry without decoding the full pixel buffer. PNG and
// JPEG use Go's header decoders; WebP/BMP use their documented file headers.
func imageInfo(raw []byte) (int, int, string, error) {
	if cfg, format, err := image.DecodeConfig(bytes.NewReader(raw)); err == nil {
		if format == "png" || format == "jpeg" {
			return cfg.Width, cfg.Height, "image/" + format, nil
		}
	}
	if len(raw) >= 54 && string(raw[:2]) == "BM" {
		dib := binary.LittleEndian.Uint32(raw[14:18])
		if dib >= 40 {
			w := int(int32(binary.LittleEndian.Uint32(raw[18:22])))
			h := int(int32(binary.LittleEndian.Uint32(raw[22:26])))
			if h < 0 {
				h = -h
			}
			return w, h, "image/bmp", nil
		}
	}
	if len(raw) >= 30 && string(raw[:4]) == "RIFF" && string(raw[8:12]) == "WEBP" {
		u24 := func(b []byte) int { return int(b[0]) | int(b[1])<<8 | int(b[2])<<16 }
		switch string(raw[12:16]) {
		case "VP8X":
			return 1 + u24(raw[24:27]), 1 + u24(raw[27:30]), "image/webp", nil
		case "VP8 ":
			if bytes.Equal(raw[23:26], []byte{0x9d, 0x01, 0x2a}) {
				return int(binary.LittleEndian.Uint16(raw[26:28]) & 0x3fff), int(binary.LittleEndian.Uint16(raw[28:30]) & 0x3fff), "image/webp", nil
			}
		case "VP8L":
			if raw[20] == 0x2f {
				b := binary.LittleEndian.Uint32(raw[21:25])
				return int(b&0x3fff) + 1, int((b>>14)&0x3fff) + 1, "image/webp", nil
			}
		}
	}
	return 0, 0, "", fmt.Errorf("must contain a readable png/jpeg/webp/bmp image header")
}

func validateLocalImagePath(path, field string) error {
	if !filepath.IsAbs(path) {
		return fmt.Errorf("%s must be an absolute path", field)
	}
	if _, ok := initImageExts[strings.ToLower(filepath.Ext(path))]; !ok {
		return fmt.Errorf("%s must be a png/jpg/jpeg/webp/bmp file", field)
	}
	st, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("%s: %w", field, err)
	}
	if !st.Mode().IsRegular() {
		return fmt.Errorf("%s must be a regular file", field)
	}
	if st.Size() > maxInitImagePathBytes {
		return fmt.Errorf("%s exceeds %d MiB limit", field, maxInitImagePathBytes>>20)
	}
	return nil
}

func resolveGenerationImages(p GenerateParams) (GenerateParams, error) {
	// Revalidate just before reading; disk inputs may have changed during
	// model loading. Clone slices before conversion to avoid caller mutation.
	if err := validateGenerationInputs(p); err != nil {
		return p, err
	}
	p.RefImages = append([]string(nil), p.RefImages...)
	p.Lora = append([]Lora(nil), p.Lora...)
	for _, target := range []struct {
		path  string
		dst   *string
		field string
	}{
		{p.InitImagePath, &p.InitImage, "init_image"}, {p.MaskImagePath, &p.MaskImage, "mask_image"}, {p.ControlImagePath, &p.ControlImage, "control_image"},
	} {
		if target.path != "" {
			raw, mime, err := readImageInput(imageInput{field: target.field, path: target.path})
			if err != nil {
				return p, err
			}
			*target.dst = "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(raw)
		}
	}
	for _, path := range p.RefImagePaths {
		raw, mime, err := readImageInput(imageInput{field: "ref_image", path: path})
		if err != nil {
			return p, err
		}
		p.RefImages = append(p.RefImages, "data:"+mime+";base64,"+base64.StdEncoding.EncodeToString(raw))
	}
	return p, nil
}

func validateLora(l Lora) error {
	if strings.TrimSpace(l.Path) == "" {
		return fmt.Errorf("lora path is required")
	}
	if math.IsNaN(l.Multiplier) || math.IsInf(l.Multiplier, 0) || l.Multiplier < -10 || l.Multiplier > 10 {
		return fmt.Errorf("lora multiplier must be finite and between -10 and 10")
	}
	return nil
}

var loraTag = regexp.MustCompile(`(?i)<lora:([^<>]+)>`)

func resolveLoraPath(path string, caps *APICapabilities, settings LoadSettings) (string, error) {
	if filepath.IsAbs(path) {
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".gguf" && ext != ".safetensors" {
			return "", fmt.Errorf("lora path must be gguf or safetensors")
		}
		st, err := os.Stat(path)
		if err != nil {
			return "", fmt.Errorf("lora path: %w", err)
		}
		if !st.Mode().IsRegular() {
			return "", fmt.Errorf("lora path must be a regular file")
		}
		for _, entry := range caps.Loras {
			if filepath.IsAbs(entry.Path) && filepath.Clean(entry.Path) == filepath.Clean(path) {
				return entry.Path, nil
			}
			if settings.LoraModelDir != "" && filepath.IsAbs(settings.LoraModelDir) {
				candidate := filepath.Join(settings.LoraModelDir, entry.Path)
				if filepath.Clean(candidate) == filepath.Clean(path) {
					return entry.Path, nil
				}
			}
		}
		return "", fmt.Errorf("absolute lora path is not in the loaded runtime's advertised catalog; configure lora_model_dir to include this file")
	}
	if filepath.Clean(path) == ".." || strings.HasPrefix(filepath.Clean(path), ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("lora path must stay within its configured directory")
	}
	for _, entry := range caps.Loras {
		if path == entry.Name || path == entry.Path {
			return entry.Path, nil
		}
	}
	if settings.LoraModelDir != "" {
		for _, suffix := range []string{"", ".safetensors", ".gguf"} {
			full := filepath.Join(settings.LoraModelDir, path+suffix)
			if found, err := resolveLoraPath(full, caps, LoadSettings{}); err == nil {
				return found, nil
			}
		}
	}
	return "", fmt.Errorf("lora %q is not in the advertised catalog or configured local LoRA directory", path)
}

func convertPromptLora(p GenerateParams, caps *APICapabilities, settings LoadSettings) (GenerateParams, error) {
	p.Lora = append([]Lora(nil), p.Lora...)
	if strings.Contains(strings.ToLower(p.NegativePrompt), "<lora:") {
		return p, fmt.Errorf("LoRA tags in negative_prompt are unsupported; use structured lora")
	}
	var convertErr error
	p.Prompt = loraTag.ReplaceAllStringFunc(p.Prompt, func(tag string) string {
		inside := tag[len("<lora:") : len(tag)-1]
		sep := strings.LastIndex(inside, ":")
		if sep < 1 {
			convertErr = fmt.Errorf("invalid LoRA prompt tag")
			return ""
		}
		weight, err := strconv.ParseFloat(inside[sep+1:], 64)
		if err != nil {
			convertErr = fmt.Errorf("invalid LoRA prompt tag multiplier")
			return ""
		}
		p.Lora = append(p.Lora, Lora{Path: inside[:sep], Multiplier: weight})
		return ""
	})
	if convertErr != nil {
		return p, convertErr
	}
	if strings.Contains(strings.ToLower(p.Prompt), "<lora:") {
		return p, fmt.Errorf("malformed LoRA prompt tag")
	}
	if len(p.Lora) > 16 {
		return p, fmt.Errorf("lora exceeds 16 entry limit")
	}
	for i, l := range p.Lora {
		if err := validateLora(l); err != nil {
			return p, err
		}
		path, err := resolveLoraPath(l.Path, caps, settings)
		if err != nil {
			return p, err
		}
		p.Lora[i].Path = path
	}
	p.Prompt = strings.TrimSpace(p.Prompt)
	return p, nil
}
