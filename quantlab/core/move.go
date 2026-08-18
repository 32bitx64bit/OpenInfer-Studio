package core

import "fmt"

// Move is a single per-tensor dtype change within a search step.
type Move struct {
	TensorName string `json:"tensorName"`
	From       DType  `json:"from"`
	To         DType  `json:"to"`
}

func (m Move) Validate() error {
	if m.TensorName == "" {
		return fmt.Errorf("move: empty tensor name")
	}
	if !m.From.Valid() || !m.To.Valid() {
		return fmt.Errorf("move %q: invalid dtype %q -> %q", m.TensorName, m.From, m.To)
	}
	if m.From == m.To {
		return fmt.Errorf("move %q: from == to (%q)", m.TensorName, m.From)
	}
	return nil
}

// ByteDelta returns the signed size change of applying the move to a tensor
// of the given element count (positive = larger).
func (m Move) ByteDelta(elements uint64) (int64, error) {
	from, ok := m.From.ExactBytes(elements)
	if !ok {
		return 0, fmt.Errorf("move %q: no geometry for %q", m.TensorName, m.From)
	}
	to, ok := m.To.ExactBytes(elements)
	if !ok {
		return 0, fmt.Errorf("move %q: no geometry for %q", m.TensorName, m.To)
	}
	return int64(to) - int64(from), nil
}

// MoveGroup is a jointly-applied set of moves whose interaction (synergy) is
// measured together, not tensor-by-tensor. KLD search commits or reverts the
// group atomically.
type MoveGroup struct {
	ID     string `json:"id"`
	Moves  []Move `json:"moves"`
	Reason string `json:"reason"`
	// JointKLD, when Measured is true, is the divergence of the artifact with
	// the whole group applied.
	JointKLD float64 `json:"jointKLD,omitempty"`
	Measured bool    `json:"measured"`
	// ByteDelta is the summed signed size change, computed at construction.
	Bytes int64 `json:"bytes"`
}

func (g MoveGroup) Validate(bank *TensorBank) error {
	if g.ID == "" {
		return fmt.Errorf("move group: empty id")
	}
	if len(g.Moves) == 0 {
		return fmt.Errorf("move group %q: no moves", g.ID)
	}
	seen := make(map[string]struct{}, len(g.Moves))
	var total int64
	for _, m := range g.Moves {
		if err := m.Validate(); err != nil {
			return err
		}
		if _, dup := seen[m.TensorName]; dup {
			return fmt.Errorf("move group %q: duplicate tensor %q", g.ID, m.TensorName)
		}
		seen[m.TensorName] = struct{}{}
		if bank != nil {
			t, ok := bank.Find(m.TensorName)
			if !ok {
				return fmt.Errorf("move group %q: unknown tensor %q", g.ID, m.TensorName)
			}
			d, err := m.ByteDelta(t.Elements)
			if err != nil {
				return err
			}
			total += d
		}
	}
	if bank != nil && g.Bytes != total {
		return fmt.Errorf("move group %q: byte delta %d, moves sum to %d", g.ID, g.Bytes, total)
	}
	return nil
}

// NewMoveGroup builds a group and computes its summed byte delta.
func NewMoveGroup(id, reason string, moves []Move, bank *TensorBank) (MoveGroup, error) {
	g := MoveGroup{ID: id, Reason: reason, Moves: moves}
	if bank != nil {
		var total int64
		for _, m := range moves {
			t, ok := bank.Find(m.TensorName)
			if !ok {
				return MoveGroup{}, fmt.Errorf("move group %q: unknown tensor %q", id, m.TensorName)
			}
			d, err := m.ByteDelta(t.Elements)
			if err != nil {
				return MoveGroup{}, err
			}
			total += d
		}
		g.Bytes = total
	}
	if err := g.Validate(bank); err != nil {
		return MoveGroup{}, err
	}
	return g, nil
}
