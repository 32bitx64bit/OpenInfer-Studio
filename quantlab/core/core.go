// Package core defines the shared domain model for the surgical quantization
// optimizer: GGUF tensor descriptors and exact-cost dtype geometry, anchor
// sets and per-tensor options, quantization profiles and selection manifests,
// measured distortion with provenance, grouped search moves, quality gates,
// and the pipeline stages that transform them.
//
// Everything here is plain data plus validation; no I/O and no external
// dependencies. Higher-level packages (tensorbank, anchor, profile, kld,
// orchestrate, state) build exclusively on these contracts.
package core
