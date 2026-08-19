package orchestrate

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"

	"quantlab/core"
)

// Artifact is one generated file with its content hash, recorded so runs are
// auditable and resumable and so cleanup is exact.
type Artifact struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

// AnchorJob is one quantize probe in an anchor batch: quantize the source at
// one anchor floor dtype so its isolated cost/quality can be measured.
type AnchorJob struct {
	Tensor  string          `json:"tensor"` // anchor name or pattern
	Floor   core.DType      `json:"floor"`
	Request QuantizeRequest `json:"request"`
}

// AnchorBatch is a deterministic set of anchor-floor quantize probes plus a
// registry of generated artifacts for hashing and cleanup.
type AnchorBatch struct {
	WorkDir   string      `json:"workDir"`
	Jobs      []AnchorJob `json:"jobs"`
	Artifacts []Artifact  `json:"artifacts,omitempty"`
}

// BuildAnchorBatch plans one quantize job per distinct anchor floor dtype
// (sorted by dtype then tensor), writing outputs into workDir with
// collision-free deterministic names. The batch is a plan: RunQuantize or an
// equivalent executes the jobs, then RecordArtifact registers outputs.
func BuildAnchorBatch(sourcePath string, anchors []core.Anchor, workDir, profileID string) (*AnchorBatch, error) {
	if sourcePath == "" || workDir == "" {
		return nil, fmt.Errorf("orchestrate: anchor batch needs source path and work dir")
	}
	type floorKey struct {
		tensor string
		floor  core.DType
	}
	seen := map[floorKey]bool{}
	var jobs []AnchorJob
	for _, a := range anchors {
		if err := a.Validate(); err != nil {
			return nil, err
		}
		if !a.MinDType.IsQuant() {
			continue
		}
		name := a.Name
		if name == "" {
			name = a.Pattern
		}
		k := floorKey{name, a.MinDType}
		if seen[k] {
			continue
		}
		seen[k] = true
		out := filepath.Join(workDir, fmt.Sprintf("anchor-%d-%s.gguf", len(jobs), a.MinDType.BaseTensorType()))
		jobs = append(jobs, AnchorJob{
			Tensor: name,
			Floor:  a.MinDType,
			Request: QuantizeRequest{
				ProfileID:  profileID,
				SourcePath: sourcePath,
				OutputPath: out,
				Type:       a.MinDType,
				// Anchor files are tensor banks. Recipe defaults are not safe here:
				// every quantizable tensor must be available at this exact dtype.
				Pure: true,
			},
		})
	}
	sort.Slice(jobs, func(i, j int) bool {
		if jobs[i].Floor != jobs[j].Floor {
			return jobs[i].Floor < jobs[j].Floor
		}
		return jobs[i].Tensor < jobs[j].Tensor
	})
	// Renumber outputs deterministically after sorting.
	for i := range jobs {
		jobs[i].Request.OutputPath = filepath.Join(workDir,
			fmt.Sprintf("anchor-%03d-%s.gguf", i, jobs[i].Floor.BaseTensorType()))
	}
	return &AnchorBatch{WorkDir: workDir, Jobs: jobs}, nil
}

// RecordArtifact hashes file at path and appends it to the batch registry.
func (b *AnchorBatch) RecordArtifact(path string) (Artifact, error) {
	a, err := HashArtifact(path)
	if err != nil {
		return Artifact{}, err
	}
	b.Artifacts = append(b.Artifacts, a)
	return a, nil
}

// Cleanup removes every recorded artifact file. Missing files are tolerated
// (idempotent); other errors are aggregated.
func (b *AnchorBatch) Cleanup() error {
	var errs []error
	for _, a := range b.Artifacts {
		if err := os.Remove(a.Path); err != nil && !os.IsNotExist(err) {
			errs = append(errs, fmt.Errorf("orchestrate: cleanup %s: %w", a.Path, err))
		}
	}
	b.Artifacts = nil
	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}

// HashArtifact streams file through SHA-256.
func HashArtifact(path string) (Artifact, error) {
	f, err := os.Open(path)
	if err != nil {
		return Artifact{}, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return Artifact{}, err
	}
	return Artifact{Path: path, SHA256: hex.EncodeToString(h.Sum(nil)), Bytes: n}, nil
}
