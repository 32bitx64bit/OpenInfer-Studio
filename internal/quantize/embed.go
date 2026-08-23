package quantize

import (
	"bytes"
	"embed"
)

//go:embed calibration/default.txt
var calibrationFS embed.FS

func defaultCalibrationBytes() []byte {
	corpus := bundledCalibrationCorpus()
	return []byte(renderCalibrationRecords(corpus.Calibration))
}

func bundledCalibrationCorpus() calibrationCorpus {
	b, err := calibrationFS.ReadFile("calibration/default.txt")
	if err != nil {
		b = []byte("OpenInfer Studio calibration text.\n")
	}
	return defaultCalibrationCorpus(bytes.TrimSpace(b))
}

func defaultCalibrationCorpus(base []byte) calibrationCorpus {
	corpus := generatedCalibrationCorpus()
	facts := factCalibrationText()
	seed := []calibrationRecord{
		{
			ID:        "openinfer-default-v1",
			Domain:    "mixed",
			Source:    "calibration/default.txt (project-authored)",
			Partition: partitionCalibration,
			Text:      string(base),
		},
		{
			ID:        "openinfer-facts-v1",
			Domain:    "facts",
			Source:    "internal/quantize/calfacts.go (project-authored)",
			Partition: partitionCalibration,
			Text:      facts,
		},
	}
	corpus.Calibration = append(seed, corpus.Calibration...)
	interleaveCalibrationCorpus(&corpus)
	return corpus
}
