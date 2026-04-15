# cuBLAS Empirical Decomposition Design

Date: 2026-04-14
Repo: `/home/sdu/zhangzhisen/GemmSim/Synperf`
Target GPU: `device 4` on an idle RTX 4090

## Goal

Reproduce the paper's cuBLAS GEMM handling strategy in a paper-faithful MVP:

- profile cuBLAS GEMM workloads on real hardware
- infer decomposition-related behavior empirically from profiling data
- normalize the inferred behavior into this repository's existing decomposition pipeline

This phase is intentionally narrower than the CUTLASS path. It does not aim to recover full internal cuBLAS tile geometry. It aims to establish a stable `profiling -> empirical decomposition` path that matches the paper's approach for closed-source libraries.

## Non-Goals

This phase does not include:

- training on cuBLAS data
- multi-GPU collection
- cuBLASLt heuristic integration as the primary method
- support for every cuBLAS kernel family
- exact per-CTA tile-coordinate reconstruction from closed-source kernels

## Scope

The MVP scope is:

- backend: `cublasLtMatmul`
- dtype: `f16`
- GPU: one idle 4090 on `device 4`
- workload: representative GEMM cases covering regular, edge, and likely split-K-triggering regimes
- output: raw profile records plus an empirical decomposition layer derived from those records

`cublasLtMatmul` is selected over classic `cublasGemmEx` because it is the more practical modern path for profiling and later extension, while still fitting the paper's profiling-based reverse-engineering methodology.

## Current Codebase Context

The repository already has:

- a mature CUTLASS decomposition and validation path
- dataset and model pipelines
- existing `cublasLt` backend skeleton files
- profiling infrastructure that can be extended

The repository does not yet have:

- a working cuBLAS/cuBLASLt benchmark binary
- cuBLAS profiling collection
- empirical decomposition rules derived from cuBLAS kernel observations

This work should therefore add a parallel cuBLAS path without disturbing the existing CUTLASS path.

## Proposed Architecture

### 1. Benchmark Binary

Add `tools/cublaslt_gemm_bench.cu`.

Responsibilities:

- run `cublasLtMatmul` for configurable `M/N/K`
- support `f16`
- measure latency with the same basic discipline as the CUTLASS benchmark path
- emit machine-readable metadata for downstream collection

The binary should expose at least:

- `M`, `N`, `K`
- `dtype`
- `device`
- average latency in microseconds

The benchmark binary should be usable both standalone and as a subprocess target for Python collection scripts.

### 2. Profiling Collection Layer

Add `src/perf_model/profiling/cublas_profile.py`.

Responsibilities:

- invoke the cuBLASLt benchmark
- invoke a profiler or trace collector to capture kernel observations
- normalize raw observations into one row-oriented profile representation

The collection layer should hide profiler-specific details from the rest of the code. The downstream code should consume a stable Python record structure, not raw profiler output.

### 3. Empirical Decomposition Layer

Add `src/perf_model/kernel_desc/cublas_empirical.py`.

Responsibilities:

- read normalized profile observations
- group observed cuBLAS kernels into empirical families
- infer decomposition-relevant attributes from profile behavior

The first version only needs to infer:

- primary kernel name or kernel family label
- whether reduction kernels are present
- number of launched kernels for one GEMM call
- launch grid and block dimensions for each observed kernel
- a normalized task-count estimate for the main GEMM kernel

The first version does not need to infer hidden internal tile coordinates. For closed-source cuBLAS, the paper's method is empirical, so the correct first milestone is family-level decomposition behavior rather than exact hidden tile geometry.

### 4. CLI Entry

Add `scripts/collect_cublas_profiles.py`.

Responsibilities:

- define a representative GEMM case list
- run collection on `device 4`
- write raw CSV records
- optionally write a normalized summary for empirical decomposition analysis

This script should stay thin and delegate logic into `profiling/` and `kernel_desc/`.

## Data Flow

The data flow is:

1. choose a GEMM case
2. run `cublasLtMatmul`
3. collect latency and kernel-level profile observations
4. normalize to raw CSV rows
5. feed rows into the empirical decomposition module
6. output decomposition summaries grouped by kernel family or rule bucket

## Raw Profile Schema

The raw CSV should include at least:

- `M`
- `N`
- `K`
- `dtype`
- `device`
- `gpu_name`
- `latency_us`
- `kernel_name`
- `kernel_index`
- `grid_x`
- `grid_y`
- `grid_z`
- `block_x`
- `block_y`
- `block_z`
- `is_reduction_kernel`
- `gemm_call_id`

If the profiler returns multiple kernels for one GEMM invocation, each kernel should become one row, keyed by the same `gemm_call_id`.

## Empirical Decomposition Output

The empirical decomposition summary should report, per GEMM case:

- main kernel family
- total kernel count
- whether auxiliary reduction kernels appear
- main-kernel grid size
- main-kernel task-count proxy

It should also support grouped analysis across many cases:

- kernel family frequency
- kernel family versus `M/N/K` regions
- reduction-kernel appearance versus `M/N/K`

## Case Selection

The first collection set should stay small and diagnostic, not exhaustive.

It should cover:

- regular aligned cases
- edge cases with non-multiple dimensions
- deeper `K`
- at least one region likely to show multi-kernel behavior

The purpose is to establish rule structure, not dataset scale.

## Error Handling

The collection path should record failure explicitly instead of silently skipping:

- benchmark launch failure
- profiler failure
- malformed or empty profile output
- unsupported cuBLASLt configuration

Failure rows should include the case metadata and an error string so later sweeps can separate unsupported cases from collector bugs.

## Verification Strategy

### Unit Tests

Add tests for:

- profile row normalization
- reduction-kernel detection
- kernel-family grouping
- task-count extraction from main-kernel grid metadata

### Real Smoke Test

Run a small real collection on `device 4` and verify:

- the benchmark executes successfully
- profile rows are emitted
- at least one case yields a stable main-kernel summary

### Success Criteria

This phase is successful if:

- cuBLASLt GEMM cases can be profiled on `device 4`
- normalized raw profile CSV is produced
- the empirical decomposition module extracts stable decomposition summaries from the profile data
- the implementation fits cleanly into the existing repository structure

## Risks and Constraints

### Profiler Choice

The largest implementation risk is the profiler interface. The MVP should prefer the simplest stable path that exposes kernel names and launch dimensions. If one profiler path is brittle, the normalization layer must make it replaceable without changing downstream decomposition logic.

### Closed-Source Boundary

cuBLAS internals are not directly visible. The design therefore intentionally stops at empirical decomposition behavior. Trying to infer full hidden tile coordinates in this phase would expand scope without adding paper-faithful value.

### GPU Isolation

This work must only use the idle 4090 on `device 4`. Collection code should allow explicit device selection and should not rely on ambient default-device behavior.

## Future Extensions

If the MVP succeeds, the next extensions are:

- larger cuBLAS profile sweeps
- support for more dtypes
- comparison between empirical cuBLAS decomposition and cuBLASLt heuristic metadata
- integration of cuBLAS empirical decomposition into dataset and training flows
