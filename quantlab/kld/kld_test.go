package kld

import "testing"

func TestInteractionValidate(t *testing.T) {
	ok := Interaction{TensorA: "a", TensorB: "b"}
	if err := ok.Validate(); err != nil {
		t.Fatalf("valid interaction rejected: %v", err)
	}
	for _, i := range []Interaction{
		{TensorA: "", TensorB: "b"},
		{TensorA: "a", TensorB: ""},
		{TensorA: "x", TensorB: "x"},
	} {
		if err := i.Validate(); err == nil {
			t.Errorf("expected error for %+v", i)
		}
	}
}

func TestSynergy(t *testing.T) {
	i := Interaction{TensorA: "a", TensorB: "b", JointDelta: 0.9, SumDelta: 0.5}
	if got := i.Synergy(); got < 0.399 || got > 0.401 {
		t.Fatalf("synergy = %v, want ~0.4", got)
	}
}
