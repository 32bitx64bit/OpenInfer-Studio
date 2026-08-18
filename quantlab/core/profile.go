package core

import "fmt"

// QuantAssignment is the per-tensor output of the profile solver.
type QuantAssignment struct {
	TensorName string `json:"tensorName"`
	Target     DType  `json:"target"`
	// BitsPerWeight is the solver's estimate used for size accounting.
	BitsPerWeight float64 `json:"bitsPerWeight"`
}

// Profile is a complete, reproducible quantization plan.
type Profile struct {
	ID          string            `json:"id"`
	BaseModel   string            `json:"baseModel"`
	Assignments []QuantAssignment `json:"assignments"`
	Anchors     []Anchor          `json:"anchors"`
	// BudgetBytes, when > 0, is the rate constraint the solver honored.
	BudgetBytes uint64 `json:"budgetBytes,omitempty"`
	// EstimatedBytes is the projected output size.
	EstimatedBytes uint64 `json:"estimatedBytes"`
}

func (p *Profile) Validate(bank *TensorBank) error {
	if p.ID == "" {
		return fmt.Errorf("profile: empty id")
	}
	if len(p.Assignments) == 0 {
		return fmt.Errorf("profile %q: no assignments", p.ID)
	}
	if bank != nil {
		for _, qa := range p.Assignments {
			if !qa.Target.Valid() {
				return fmt.Errorf("profile %q: tensor %q has invalid target %q", p.ID, qa.TensorName, qa.Target)
			}
			if _, ok := bank.Find(qa.TensorName); !ok {
				return fmt.Errorf("profile %q: unknown tensor %q", p.ID, qa.TensorName)
			}
			if qa.BitsPerWeight <= 0 {
				return fmt.Errorf("profile %q: tensor %q has non-positive bpw", p.ID, qa.TensorName)
			}
		}
		for _, a := range p.Anchors {
			if err := a.Validate(); err != nil {
				return err
			}
		}
	}
	return nil
}

// SelectionManifest is the frozen per-tensor selection produced by the solve
// stage and consumed by quantize: exactly one option per tensor, with exact
// total byte cost. Unlike Profile (a planning artifact), a manifest is the
// assembly contract.
type SelectionManifest struct {
	ProfileID string `json:"profileID"`
	// SourceSHA is the hex digest of the primary GGUF assembly reads
	// payloads from (reconstructed/folded when those stages ran, else
	// TensorBank.SHA256).
	SourceSHA string         `json:"sourceSHA,omitempty"`
	Options   []TensorOption `json:"options"`
	// TotalBytes is the exact sum of Options[].Bytes.
	TotalBytes uint64 `json:"totalBytes"`
}

func (m *SelectionManifest) Validate(bank *TensorBank) error {
	if m.ProfileID == "" {
		return fmt.Errorf("manifest: empty profile id")
	}
	if len(m.Options) == 0 {
		return fmt.Errorf("manifest %q: no options", m.ProfileID)
	}
	seen := make(map[string]struct{}, len(m.Options))
	var total uint64
	for _, o := range m.Options {
		if err := o.Validate(); err != nil {
			return err
		}
		if _, dup := seen[o.TensorName]; dup {
			return fmt.Errorf("manifest %q: duplicate tensor %q", m.ProfileID, o.TensorName)
		}
		seen[o.TensorName] = struct{}{}
		// Byte cost must match geometry exactly.
		if bank != nil {
			t, ok := bank.Find(o.TensorName)
			if !ok {
				return fmt.Errorf("manifest %q: unknown tensor %q", m.ProfileID, o.TensorName)
			}
			want, ok := o.Target.ExactBytes(t.Elements)
			if ok && want != o.Bytes {
				return fmt.Errorf("manifest %q: tensor %q byte cost %d, geometry says %d",
					m.ProfileID, o.TensorName, o.Bytes, want)
			}
		}
		total += o.Bytes
	}
	if m.TotalBytes != total {
		return fmt.Errorf("manifest %q: total %d does not match option sum %d", m.ProfileID, m.TotalBytes, total)
	}
	return nil
}

// ManifestFor builds and validates a manifest from a profile and bank.
// Every bank tensor must be covered exactly once.
func ManifestFor(p *Profile, bank *TensorBank) (*SelectionManifest, error) {
	if p == nil || bank == nil {
		return nil, fmt.Errorf("manifest: nil profile or bank")
	}
	covered := make(map[string]bool, len(p.Assignments))
	opts := make([]TensorOption, 0, len(p.Assignments))
	for _, qa := range p.Assignments {
		t, ok := bank.Find(qa.TensorName)
		if !ok {
			return nil, fmt.Errorf("manifest: unknown tensor %q", qa.TensorName)
		}
		b, ok := qa.Target.ExactBytes(t.Elements)
		if !ok {
			return nil, fmt.Errorf("manifest: no geometry for %q", qa.Target)
		}
		opts = append(opts, TensorOption{TensorName: qa.TensorName, Target: qa.Target, Bytes: b})
		covered[qa.TensorName] = true
	}
	for _, t := range bank.Tensors {
		if !covered[t.Name] {
			return nil, fmt.Errorf("manifest: profile %q missing tensor %q", p.ID, t.Name)
		}
	}
	m := &SelectionManifest{ProfileID: p.ID, SourceSHA: bank.SHA256, Options: opts}
	for _, o := range opts {
		m.TotalBytes += o.Bytes
	}
	if err := m.Validate(bank); err != nil {
		return nil, err
	}
	return m, nil
}
