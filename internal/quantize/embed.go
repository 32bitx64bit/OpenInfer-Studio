package quantize

import "embed"

//go:embed calibration/default.txt
var calibrationFS embed.FS

func defaultCalibrationBytes() []byte {
	b, err := calibrationFS.ReadFile("calibration/default.txt")
	if err != nil {
		return []byte("OpenInfer Studio calibration text.\n")
	}
	return b
}
