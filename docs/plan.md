# Plan: Kernel Fusion Analysis & Benchmarking for FIDESlib

## Context

The goal is to (1) identify which existing GPU kernels in FIDESlib can be fused into a single kernel pass, and (2) measure the performance benefit using the existing benchmark infrastructure by comparing `time(kernel A) + time(kernel B)` vs. `time(fused kernel A+B)`.

This is a two-phase effort: **analysis** (identify candidates) and **benchmarking** (measure and compare). No new fused kernel implementations are required upfront — we first use the existing benchmark suite to quantify the cost of running ops sequentially, then identify where fusion already exists or would be most impactful.

---

## Phase 1: Understand What Fusion Already Exists

FIDESlib already has substantial kernel fusion. Before writing any new code, document the existing fusion so you know what gap remains.

### Already-fused kernels (do NOT re-implement these)

| Fused kernel | Mode/function | File |
|---|---|---|
| NTT + Rescale | `NTT_<T, _, _, NTT_RESCALE>`, called via `NTT_rescale_fused()` | `src/CKKS/Limb.cu:367` |
| NTT + ModDown | `NTT_<T, _, _, NTT_MODDOWN>`, called via `NTT_moddown_fused()` | `src/CKKS/Limb.cu:418` |
| NTT + Plaintext Mult | `NTT_<T, _, _, NTT_MULTPT>`, called via `NTT_multpt_fused()` | `src/CKKS/Limb.cu:471` |
| INTT + KSK multiply | `INTT_MULT_AND_SAVE` / `INTT_MULT_AND_ACC` modes | `src/CKKS/Limb.cu:658, 725` |
| Rotation + KSK dot | `hoistedRotateDotKSK_2_` kernel | `src/CKKS/ElemenwiseBatchKernels.cu:299` |
| KSK dot product | `fusedDotKSK_2_` kernel | `src/CKKS/ElemenwiseBatchKernels.cu:221` |
| mult + add + add | `mult1AddMult23Add4_` kernel | `src/CKKS/ElemenwiseBatchKernels.cu:17` |
| binomial square fold | `binomial_square_fold_` kernel | `src/CKKS/ElemenwiseBatchKernels.cu:134` |

### Unfused candidates worth investigating

The most impactful unfused sequences are:

| Sequence | Why it's a candidate | Where called |
|---|---|---|
| **MultPt → Rescale** | Plaintext mult raises noise level; rescale is the natural next step. Both are memory-bandwidth-bound on the same RNS arrays. | `Ciphertext::multPt()` then `Ciphertext::rescale()` — measured separately in `PlaintextMultiplicationBenchmarks.cu` |
| **Ct×Ct Mult → Rescale** | After every ciphertext multiplication, rescaling is mandatory. Same data dependency. | `Ciphertext::mult()` then `Ciphertext::rescale()` |
| **Add → Rescale** | Less common, but possible when noise budgets require it. | `CiphertextAddBenchmarks.cu` + `PlaintextMultiplicationBenchmarks.cu:Rescale` |
| **ModUp → ModDown** | These are already benchmarked together (`CiphertextModUpModDown`) at ciphertext level and (`RNSPolyStandardModUpModDown`) at RNS level. Use existing benchmark as the template. | `RNSPolyModUpModDownBenchmarks.cu` |

---

## Phase 2: Use Existing Benchmarks to Measure Baseline Sequential Cost

Before writing any new code, run the benchmarks that already measure the relevant operations in isolation. This gives the baseline `time(A) + time(B)` numbers.

### Step 1: Build and run the benchmark binary

```bash
cd /path/to/FIDESlib
cmake --build build -j
cd build
./fideslib-bench --benchmark_filter="MultPlaintext|Rescale|CiphertextMultiplication|CiphertextModUp|CiphertextModDown|RNSPolyStandardModDown|RNSPolyModUp"
```

Record the per-operation wall-clock times for a fixed parameter set (e.g., parameter index 3 = `gparams64_14`, level 0).

### Step 2: Run the already-combined benchmarks

```bash
./fideslib-bench --benchmark_filter="CiphertextModUpModDown|RNSPolyStandardModUpModDown"
```

These benchmarks already measure A+B as one block. Compare against the sum of the individual timings from Step 1.

If `time(A+B combined) < time(A alone) + time(B alone)`, kernel-launch overhead and memory round-trips are real costs — fusion is worthwhile.

---

## Phase 3: Add a New Fused Benchmark for MultPt+Rescale

This is the highest-value fusion candidate because:
- `multPt` and `rescale` are **always** called back-to-back in practice
- Both are RNS-polynomial-level operations on the same ciphertext limbs
- The existing code already has `NTT_multpt_fused` (NTT stage) — the rescale stage could extend this

### 3a: Add a combined benchmark (no new kernel needed yet)

Create `bench/MultPtRescaleBenchmarks.cu` following the pattern in `bench/PlaintextMultiplicationBenchmarks.cu`. Measure `multPt() + rescale()` sequentially in one timing block.

**File to create:** `bench/MultPtRescaleBenchmarks.cu`

**Template:** Copy structure from `bench/CiphertextModUpModDownBenchmarks.cu` (uses `UseManualTime()` pattern for precise isolation).

```cpp
// Measure A alone (multPt)
auto start = chrono::high_resolution_clock::now();
GPUct1.multPt(GPUpt2, false);
cudaDeviceSynchronize();
auto mid = chrono::high_resolution_clock::now();

// Measure A+B combined (multPt + rescale)
GPUct1.multPt(GPUpt2, false);
GPUct1.rescale();
cudaDeviceSynchronize();
auto end = chrono::high_resolution_clock::now();
```

**Wire into CMakeLists:** Add the new `.cu` file to the `fideslib-bench` target in `CMakeLists.txt` (look for the existing list of bench sources).

### 3b: Compare numbers

After running, you get:
- `T_multPt` — time for multPt alone
- `T_rescale` — time for rescale alone
- `T_combined` — time running both sequentially in one benchmark (no separate kernel launch overhead)

The "overhead" from unfused execution = `T_multPt + T_rescale - T_combined`. If this is >5%, a fused kernel is worth implementing.

---

## Phase 4: Interpret Results & Decide Whether to Write a Fused Kernel

| Outcome | Interpretation | Action |
|---|---|---|
| `T_combined ≈ T_A + T_B` | No kernel-launch overhead; ops are compute-bound; fusion won't help | Document result; no new kernel needed |
| `T_combined < T_A + T_B` by >5% | Kernel launch / memory round-trip overhead is real | Proceed to implement fused kernel |
| ModUp+ModDown combined is already faster | Confirms pattern seen in RNSPolyModUpModDown | Use this as proof-of-concept template |

---

## Critical Files

| File | Role |
|---|---|
| `src/CKKS/Limb.cu` | Host wrappers that launch all kernel sequences; best place to add fused host functions |
| `src/CKKS/ElemenwiseBatchKernels.cu` | All fused `__global__` kernels; add new fused CUDA kernels here |
| `src/NTT.cu` / `src/NTT.cuh` | NTT kernel with template mode parameter; existing fusion modes here |
| `src/CKKS/Rescale.cu` | `SwitchModulus` kernel; needed to understand rescale's memory pattern |
| `bench/PlaintextMultiplicationBenchmarks.cu` | MultPt and Rescale measured individually (use as baseline reference) |
| `bench/CiphertextModUpModDownBenchmarks.cu` | Gold-standard example of A / B / A+B benchmark pattern |
| `bench/RNSPolyModUpModDownBenchmarks.cu` | Lower-level version of same pattern; template for new bench |
| `bench/Benchmark.cuh` | Fixture classes (`GeneralFixture`, `FIDESlibFixture`) and config macros |
| `bench/Benchmark.cu` | 28 parameter configurations; pick parameter index 3 (gparams64_14) for initial tests |
| `CMakeLists.txt` | Add new benchmark source files here |

---

## Verification

1. Build succeeds: `cmake --build build -j` produces `fideslib-bench`
2. New benchmark runs without error: `./fideslib-bench --benchmark_filter="MultPtRescale"`
3. Numbers are plausible: `T_combined <= T_multPt + T_rescale` (combined should never be worse)
4. Compare against baseline: `./fideslib-bench --benchmark_filter="MultPlaintext|Rescale|CiphertextModUpModDown"` and verify same parameter set
5. If fusion is implemented: correctness test via `./fideslib-test --gtest_filter=*multPt*` or relevant OpenFHE interface test
