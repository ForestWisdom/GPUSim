# 4090 CUTLASS Training Data Design

## Goal

Collect a first real training dataset for the current GEMM-only MVP on the local RTX 4090, then convert it into processed `f_*` features and train a baseline latency model.

This design intentionally keeps the backend and kernel family narrow so the first dataset answers one question cleanly:

Can the current `decomposer -> scheduler -> features -> latency model` pipeline learn from real CUTLASS Tensor Core GEMM measurements on one GPU?

## Scope

In scope:

- GPU: RTX 4090
- Backend: CUTLASS
- Kernel family: Tensor Core GEMM
- Dtype: `f16`
- Threadblock shape: `128x128x32`
- Swizzles: `Identity`, `Identity2`, `Identity4`
- Split-K: `1`, `2`
- Real latency collection via the local CUTLASS benchmark binary
- Raw profile CSV generation
- Processed dataset generation with `f_*` features plus `latency_us`
- One baseline MLP training run with validation metrics

Out of scope for this phase:

- cuBLASLt heuristic collection
- More threadblock shapes
- SIMT kernels
- Multi-GPU training data
- Attention or fused kernels
- Nsight Compute collection as part of the bulk dataset path

## Why This Scope

The current codebase has already validated the CUTLASS task decomposition path reasonably well. The bottleneck has shifted from decomposition correctness to real data availability.

Adding more kernel families or more threadblock shapes now would mix together too many sources of variance:

- benchmark legality and alignment constraints
- decomposition assumptions
- feature calibration errors
- model capacity limits

The first dataset should isolate those factors as much as possible. Fixing the GPU, dtype, and threadblock shape keeps the collected data interpretable while still allowing enough variation from `M/N/K`, `split_k`, and swizzle.

## Data Collection Strategy

### Sampling Space

Each raw sample is defined by:

- `M`
- `N`
- `K`
- `split_k_slices`
- `swizzle`

The benchmark configuration remains fixed at:

- `dtype=f16`
- `threadblock_shape=(128, 128, 32)`
- `warp_shape` and instruction shape implied by the current CUTLASS benchmark path
- `device=4090`

### Parameter Coverage

The dataset must include both regular and edge cases.

`M` and `N` coverage:

- aligned points such as `128`, `256`, `384`, `512`
- off-by-small-tail points such as `129`, `130`, `255`, `257`
- medium and larger points so task count varies significantly

`K` coverage:

- aligned points such as `32`, `64`, `96`, `128`, `256`
- tail points such as `33`, `65`, `127`, `129`, `255`

`split_k_slices` coverage:

- `1`
- `2`

`swizzle` coverage:

- `Identity`
- `Identity2`
- `Identity4`

### Sampling Method

Do not run the full Cartesian product of every candidate value. That would waste time on redundant neighboring points and increase failures from benchmark-invalid combinations without increasing training value proportionally.

Instead, generate a layered sample set:

1. core aligned points
2. edge-tail points
3. mixed aligned-edge points
4. larger points that increase CTA count and scheduling imbalance

Then cross these samples with `split_k` and swizzle values.

Target scale for the first pass:

- approximately `1k-5k` valid samples after filtering failures

## Raw Profile Format

The raw CSV should live under `data/raw/profiles/` and contain one row per attempted benchmark run.

Required fields:

- `gpu_name`
- `device`
- `kernel_name`
- `dtype`
- `threadblock_m`
- `threadblock_n`
- `threadblock_k`
- `swizzle`
- `split_k_slices`
- `M`
- `N`
- `K`
- `latency_us`
- `iterations`
- `warmup`
- `status`

Optional but useful fields:

- `task_count`
- `swizzle_log_tile`
- `grid_m`
- `grid_n`
- `grid_k`
- `error_message`

`status` must distinguish at least:

- `ok`
- `rejected`
- `runtime_error`

Only `ok` rows move forward into processed dataset generation.

## Processed Dataset Format

The processed CSV should live under `data/processed/` and contain:

- `latency_us`
- problem metadata
- kernel metadata used for decomposition
- flat feature columns `f_*`

The processed dataset path should preserve `split_k_slices` and `swizzle` from the raw profile row rather than silently using only the YAML default kernel config.

This is important because the first training dataset deliberately varies both fields.

## Implementation Plan For The Data Path

### 1. Real profile collection

Replace the current `scripts/collect_profiles.py` stub with a real CLI that:

- chooses the local device
- generates the parameter sweep
- runs the CUTLASS benchmark through `profiling.runner`
- records both successful and failed cases
- writes a raw CSV incrementally

The script must checkpoint progress to disk as it runs so a long collection job can be resumed without losing completed samples.

### 2. Dataset building

Extend `scripts/build_dataset.py` and the dataset builder path so each record can override:

- `split_k_slices`
- `swizzle`

This avoids forcing one kernel YAML per swizzle or split-K mode and keeps the raw-to-processed pipeline straightforward.

### 3. Training entrypoint

Use the existing training pipeline and checkpoint format, which already supports:

- train/val split
- feature normalization
- best-model selection by validation loss

Add no new modeling complexity in this phase beyond ensuring the processed dataset can flow into a real baseline experiment.

## Failure Handling

Expected failures are normal in the raw collection stage because some combinations will be rejected or produce runtime errors depending on benchmark legality.

Rules:

- keep failed rows in raw CSV with status and error message
- exclude failed rows from processed dataset generation
- do not silently drop rows without recording why

This makes the collection run auditable and easier to debug.

## Verification

The work is complete for this phase when the following all succeed:

1. A raw 4090 profile CSV is produced from real benchmark runs.
2. A processed dataset CSV is generated from successful rows.
3. A baseline training run completes and writes a checkpoint.
4. The training CLI reports validation metrics including RMSE and MAPE.

Recommended verification commands after implementation:

```bash
python scripts/collect_profiles.py --help
python scripts/build_dataset.py --help
python scripts/train_mlp.py --help
python -m pytest -q
```

## Risks

### Benchmark legality

Some edge combinations may still be rejected or fail at runtime even within the narrowed kernel family.

Mitigation:

- record status per row
- filter failed rows for processed dataset generation

### Data imbalance

If aligned points dominate, the model may underfit edge behavior.

Mitigation:

- explicitly include tail-heavy and mixed edge cases in the sweep generator

### Throughput of collection

Large sweeps may take a long time if each run launches the benchmark process separately.

Mitigation:

- write the collector so it saves progress incrementally
- start with a moderate first pass and scale after measuring runtime

## Recommendation

Proceed with the single-GPU, single-kernel-family dataset first. It is the highest-signal next step and keeps the data generation problem well-bounded.
