package pipeline

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"quantlab/core"
	"quantlab/mtp"
	"quantlab/tensorbank"
)

func (e *Engine) jobBPW() float64 {
	if e.Run == nil {
		return 0
	}
	if e.Run.Config.TargetBPW > 0 {
		return e.Run.Config.TargetBPW
	}
	return mtp.ImpliedBPW(e.Run.Bank, e.Run.Config.BudgetBytes)
}

// stageDropMTP removes fused Multi-Token Prediction / NextN tensors from
// the latest payload GGUF for Q2-class jobs and patches nextn_predict_layers
// to 0 so llama.cpp loads a vanilla trunk. BudgetBytes is left unchanged so
// the saved bytes accrue to remaining tensors. Dedicated MTP sidecars are
// not stripped.
func (e *Engine) stageDropMTP(ctx context.Context) error {
	if e.Run == nil || e.Run.Bank == nil {
		return nil
	}
	if !mtp.ShouldDrop(e.jobBPW()) {
		return nil
	}
	if e.DryRun {
		e.printf("plan: drop fused MTP/NextN tensors from payload\n")
		return nil
	}

	outPath := filepath.Join(e.workDir(), "source-no-mtp.gguf")
	nBefore := len(e.Run.Bank.Tensors)
	if p := e.Extra.MTPStrippedPath; p != "" {
		if _, err := os.Stat(p); err == nil {
			if err := e.resumeStrippedMTP(p); err == nil {
				e.logMTPDrop(nBefore, p)
				return nil
			}
		}
	}
	if _, err := os.Stat(outPath); err == nil {
		if err := e.resumeStrippedMTP(outPath); err == nil {
			e.logMTPDrop(nBefore, outPath)
			return nil
		}
	}

	srcPath := e.weightPayloadSource()
	src, err := tensorbank.OpenSource(srcPath)
	if err != nil {
		return fmt.Errorf("mtp: open payload: %w", err)
	}
	defer src.Close()
	f, err := tensorbank.Parse(src)
	if err != nil {
		return fmt.Errorf("mtp: parse payload: %w", err)
	}
	drop, ok := mtp.Select(e.Run.Bank, f.KVs, f.Architecture)
	if !ok {
		return nil
	}

	keep := make(map[string]struct{}, len(e.Run.Bank.Tensors)-len(drop))
	skip := make(map[string]struct{}, len(drop))
	for _, name := range drop {
		skip[name] = struct{}{}
	}
	for _, t := range e.Run.Bank.Tensors {
		if _, gone := skip[t.Name]; !gone {
			keep[t.Name] = struct{}{}
		}
	}
	kvs := mtp.ZeroNextnLayers(append([]tensorbank.KV(nil), f.KVs...), f.Architecture)
	if err := os.MkdirAll(e.workDir(), 0o755); err != nil {
		return err
	}
	if err := tensorbank.TrimWithMetadata(ctx, srcPath, keep, outPath, kvs, e.progressFunc(core.StageAssemble, "drop MTP")); err != nil {
		return fmt.Errorf("mtp: trim: %w", err)
	}
	if err := e.adoptStrippedMTP(outPath); err != nil {
		return err
	}
	e.printf("  mtp: dropped %d fused nextn tensors -> %s\n", len(drop), outPath)
	return nil
}

func (e *Engine) logMTPDrop(nBefore int, path string) {
	n := 0
	if e.Run.Bank != nil {
		n = nBefore - len(e.Run.Bank.Tensors)
	}
	if n < 0 {
		n = 0
	}
	e.printf("  mtp: dropped %d fused nextn tensors -> %s\n", n, path)
}

// resumeStrippedMTP binds an existing stripped GGUF. If the bank already
// lacks those tensors, this is a no-op besides ensuring Extra.MTPStrippedPath.
func (e *Engine) resumeStrippedMTP(path string) error {
	s, err := tensorbank.OpenSource(path)
	if err != nil {
		return err
	}
	defer s.Close()
	f, err := tensorbank.Parse(s)
	if err != nil {
		return err
	}
	stripped := bankFromGGUF(path, f)
	if drop, ok := mtp.Select(stripped, f.KVs, f.Architecture); ok {
		return fmt.Errorf("mtp: %s still has %d nextn tensors", path, len(drop))
	}
	if sameTensorNames(e.Run.Bank, stripped) {
		e.Extra.MTPStrippedPath = path
		return e.saveExtra()
	}
	return e.adoptStrippedMTP(path)
}

func bankFromGGUF(path string, f *tensorbank.File) *core.TensorBank {
	b := &core.TensorBank{
		SourcePath:    path,
		ModelID:       f.ModelID,
		Alignment:     uint64(f.Alignment),
		KVMetadataLen: uint64(len(f.KVBytes)),
		Arch:          f.Architecture,
	}
	for _, t := range f.Tensors {
		b.Tensors = append(b.Tensors, core.TensorDesc{
			Name: t.Name, DType: t.DType, Shape: append([]uint64(nil), t.Shape...),
			Offset: t.RelOffset, Length: t.Length, Elements: t.Elements,
		})
	}
	return b
}

func sameTensorNames(a, b *core.TensorBank) bool {
	if a == nil || b == nil || len(a.Tensors) != len(b.Tensors) {
		return false
	}
	seen := make(map[string]struct{}, len(a.Tensors))
	for _, t := range a.Tensors {
		seen[t.Name] = struct{}{}
	}
	for _, t := range b.Tensors {
		if _, ok := seen[t.Name]; !ok {
			return false
		}
	}
	return true
}

func (e *Engine) adoptStrippedMTP(path string) error {
	s, err := tensorbank.OpenSource(path)
	if err != nil {
		return err
	}
	defer s.Close()
	modelID := ""
	if e.Run.Bank != nil {
		modelID = e.Run.Bank.ModelID
	}
	bank, err := tensorbank.NewAssembler().Assemble(s, path, modelID)
	if err != nil {
		return fmt.Errorf("mtp: assemble stripped: %w", err)
	}
	pf, err := tensorbank.Parse(s)
	if err != nil {
		return fmt.Errorf("mtp: reparse stripped: %w", err)
	}
	if drop, ok := mtp.Select(bank, pf.KVs, pf.Architecture); ok {
		return fmt.Errorf("mtp: stripped file still has %d nextn tensors", len(drop))
	}
	e.Run.Bank = bank
	e.Extra.MTPStrippedPath = path
	e.Extra.PayloadSHA = ""
	e.Extra.PayloadSHAPath = ""
	if err := e.saveExtra(); err != nil {
		return fmt.Errorf("mtp: persist extra config: %w", err)
	}
	return nil
}
