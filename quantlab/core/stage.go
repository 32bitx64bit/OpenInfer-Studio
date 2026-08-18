package core

// Stage enumerates the resumable pipeline stages in execution order.
type Stage string

const (
	StageAssemble Stage = "assemble" // build TensorBank from source GGUF
	StageAnchor   Stage = "anchor"   // resolve anchor set
	StageSolve    Stage = "solve"    // candidate profile solver
	StageQuantize Stage = "quantize" // llama-quantize execution
	StageEvaluate Stage = "evaluate" // perplexity / KLD measurement
	StageSearch   Stage = "search"   // interaction-aware refinement
	StageEmit     Stage = "emit"     // final artifact + report
)

// StageOrder defines the canonical pipeline progression.
var StageOrder = []Stage{StageAssemble, StageAnchor, StageSolve, StageQuantize, StageEvaluate, StageSearch, StageEmit}

// StageIndex returns the ordinal of s in StageOrder, or -1.
func StageIndex(s Stage) int {
	for i, v := range StageOrder {
		if v == s {
			return i
		}
	}
	return -1
}

func (s Stage) Valid() bool { return StageIndex(s) >= 0 }
