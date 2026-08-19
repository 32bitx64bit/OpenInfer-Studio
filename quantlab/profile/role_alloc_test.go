package profile

import (
	"fmt"
	"strings"
	"testing"

	"quantlab/anchor"
	"quantlab/core"
)

// TestRoleAllocationAt3_5BPW locks Unsloth Dynamic 3.0 lever #2: named
// keepers (embed / output / attn_v) hold bits, dull FFN is harvested, and
// Q6_K is not painted onto every ffn_down. Heuristic path, one stock ggml
// type per GGUF name.
func TestRoleAllocationAt3_5BPW(t *testing.T) {
	const layers = 8
	const mid = 3 // mid-depth; first two are 0,1 and last two are 6,7
	bank := &core.TensorBank{SourcePath: "/synth.gguf", ModelID: "role-alloc"}
	// attn q/v/out share an element count; SwiGLU up/down (and gate) are 4×.
	// token_embd / output are smaller than one FFN tensor.
	bank.Tensors = append(bank.Tensors,
		weightTD("token_embd.weight", 256, 512),
		weightTD("output.weight", 256, 512),
	)
	for i := 0; i < layers; i++ {
		p := fmt.Sprintf("blk.%d.", i)
		bank.Tensors = append(bank.Tensors,
			weightTD(p+"attn_q.weight", 256, 256),
			weightTD(p+"attn_v.weight", 256, 256),
			weightTD(p+"attn_output.weight", 256, 256),
			weightTD(p+"ffn_gate.weight", 256, 1024),
			weightTD(p+"ffn_up.weight", 256, 1024),
			weightTD(p+"ffn_down.weight", 256, 1024),
		)
	}

	set, err := anchor.Derive(bank, nil, anchor.PolicyForBPW(3.5))
	if err != nil {
		t.Fatal(err)
	}
	cands := []core.DType{
		core.DTypeQ6_K, core.DTypeQ5_K_T, core.DTypeQ4_K_T, core.DTypeQ3_K,
		core.DTypeQ2_K, core.DTypeIQ2_XXS,
	}
	res, err := Solve(Request{
		Bank:       bank,
		Anchors:    set,
		Candidates: cands,
		TargetBPW:  3.5,
	})
	if err != nil {
		t.Fatal(err)
	}

	assign := map[string]core.DType{}
	for _, qa := range res.Profile.Assignments {
		assign[qa.TensorName] = qa.Target
	}
	dump := func() string {
		var b strings.Builder
		for _, td := range bank.Tensors {
			fmt.Fprintf(&b, "  %s = %s\n", td.Name, assign[td.Name])
		}
		return b.String()
	}

	emb := assign["token_embd.weight"]
	out := assign["output.weight"]
	midV := assign[fmt.Sprintf("blk.%d.attn_v.weight", mid)]
	midUp := assign[fmt.Sprintf("blk.%d.ffn_up.weight", mid)]
	midGate := assign[fmt.Sprintf("blk.%d.ffn_gate.weight", mid)]

	if anchor.Rank(emb) >= anchor.Rank(midUp) {
		t.Errorf("token_embd %s not higher fidelity than mid ffn_up %s\n%s", emb, midUp, dump())
	}
	if anchor.Rank(out) >= anchor.Rank(midUp) {
		t.Errorf("output %s not higher fidelity than mid ffn_up %s\n%s", out, midUp, dump())
	}
	if anchor.Rank(midV) >= anchor.Rank(midUp) {
		t.Errorf("mid attn_v %s not higher fidelity than mid ffn_up %s\n%s", midV, midUp, dump())
	}

	nDown, nDownQ6 := 0, 0
	for i := 0; i < layers; i++ {
		nDown++
		if assign[fmt.Sprintf("blk.%d.ffn_down.weight", i)] == core.DTypeQ6_K {
			nDownQ6++
		}
	}
	if nDownQ6 == nDown {
		t.Errorf("every ffn_down is Q6_K (the Q4_K_M reflex the policy must not replay)\n%s", dump())
	}

	q5 := core.DTypeQ5_K_T
	if anchor.Rank(midGate) <= anchor.Rank(q5) && anchor.Rank(midUp) <= anchor.Rank(q5) {
		t.Errorf("mid ffn_gate %s and ffn_up %s are both at-or-above Q5; want a harvested mid FFN\n%s",
			midGate, midUp, dump())
	}
}
