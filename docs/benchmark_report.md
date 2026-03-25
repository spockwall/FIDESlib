# FIDESlib Benchmark Suite — Detailed Reference

## 0. Glossary

### Google Benchmark Framework Terms

- **`benchmark::Fixture`**
  - The base class provided by Google Benchmark for stateful benchmarks. A fixture is a C++ class that holds shared setup state (GPU contexts, crypto keys, allocated buffers) that is initialized once per benchmark run, not once per iteration. All FIDESlib benchmarks inherit from either `GeneralFixture` or `FIDESlibFixture`, both of which extend `benchmark::Fixture`.

- **`BENCHMARK_DEFINE_F(FixtureClass, BenchmarkName)`**
  - A macro that declares a benchmark function attached to a fixture class. It expands to a method on the fixture class with signature `void BenchmarkName(benchmark::State& state)`. The function body contains the actual work to be timed, including the `for (auto _ : state)` loop. Defining a benchmark does not register or run it — that requires `BENCHMARK_REGISTER_F`.

- **`BENCHMARK_REGISTER_F(FixtureClass, BenchmarkName)`**
  - A macro that registers a previously defined benchmark with the Google Benchmark runner and configures how it is parameterized. Must be called after `BENCHMARK_DEFINE_F`. Returns a pointer to a `benchmark::internal::Benchmark` object that can be chained with configuration methods like `->ArgsProduct(...)` and `->UseManualTime()`.

- **`benchmark::State`**
  - The object passed into every benchmark function as `state`. It controls the benchmark loop, carries parameter values, and accepts timing data. Key members:
    - `state.range(N)` — reads the Nth parameter value for the current run
    - `state.counters["name"] = value` — attaches a labelled counter to the output row
    - `state.SetIterationTime(seconds)` — records manual timing for one iteration
    - `state.SkipWithMessage("reason")` — skips this parameter combination with a message

- **`for (auto _ : state)`**
  - The benchmark iteration loop. Google Benchmark controls how many times this runs (enough iterations to get a stable measurement). In automatic timing mode, the entire body is timed. In manual timing mode, only the region bracketed by `SetIterationTime()` counts.

- **`->ArgsProduct({list0, list1, ...})`**
  - Configures the benchmark to run once for every combination of values across the provided lists (Cartesian product). Each combination is a separate row in the output. Values are accessed inside the benchmark via `state.range(0)`, `state.range(1)`, etc., in the order the lists are given.

- **`->UseManualTime()`**
  - Tells Google Benchmark to use the time reported via `state.SetIterationTime()` instead of measuring wall-clock time automatically. Required when the benchmark loop contains setup or teardown code that must be excluded from the measurement (e.g., restoring a ciphertext to its pre-operation state after each iteration).

- **`state.SetIterationTime(elapsed.count())`**
  - Reports the elapsed time (in seconds, as a `double`) for one iteration when using manual timing. Typically called with a `std::chrono::duration` measured around only the GPU operation being benchmarked.

- **`state.counters["key"] = value`**
  - Attaches an arbitrary named counter to the benchmark output row. Used in FIDESlib to echo parameter values (`p_batch`, `p_limbs`, `p_ntt`) as human-readable columns alongside the timing result. Does not affect measurement.

- **`state.SkipWithMessage("message")`**
  - Marks the current benchmark run as skipped and records the message in output. Used in FIDESlib to skip parameter combinations where the requested RNS level exceeds the parameter set's `multDepth`.

- **`CudaCheckErrorMod`**
  - A FIDESlib macro (not Google Benchmark) that calls `cudaDeviceSynchronize()` and asserts no CUDA error occurred. Used between iterations to catch GPU errors that would otherwise be invisible due to CUDA's asynchronous execution model.

- **`cudaDeviceSynchronize()`**
  - A CUDA API call that blocks the CPU until all previously launched GPU kernels on the current device have completed. Required before stopping a manual timer to ensure the GPU work is actually finished and not just enqueued.

---

### Benchmark Structure Terms

- **Fixture**
  - A class that holds state shared across all iterations of a benchmark. In FIDESlib, fixtures hold the OpenFHE `CryptoContext`, key pairs, and FIDESlib parameter structs. The fixture's `SetUp()` method runs before the benchmark loop starts; `TearDown()` runs after. Fixtures allow expensive one-time setup (context creation, key generation) to be excluded from the timed region.

- **Iteration**
  - One execution of the `for (auto _ : state)` body. Google Benchmark runs many iterations and reports the mean time per iteration. The number of iterations is chosen automatically to achieve a stable measurement (typically targeting ~0.5–1.0 second of total benchmark time).

- **Run / Parameter combination**
  - One specific set of argument values from `ArgsProduct`. A benchmark with `ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})` produces 4 × 1 × 1 × 31 = 124 runs, each appearing as a separate row in output.

- **Warm-up**
  - The first few iterations where GPU caches are cold and JIT compilation may occur. Google Benchmark discards these automatically. Some FIDESlib benchmarks (notably in `PlaintextMultiplicationBenchmarks.cu`) implement a manual warm-up loop that doubles iteration count until elapsed time exceeds 1 second.

- **Automatic timing**
  - Default mode. Google Benchmark wraps the entire `for (auto _ : state)` body with a high-resolution timer. Simple to use but cannot exclude teardown code inside the loop.

- **Manual timing (`UseManualTime`)**
  - The benchmark code itself measures elapsed time and calls `state.SetIterationTime(t)`. Allows precise isolation of the GPU operation from surrounding restore/reset code. Used whenever the benchmark loop must modify state after the timed operation (e.g., calling `grow()` to restore a ciphertext level).

- **`state.range(N)`**
  - Reads the Nth value from the current parameter combination. Index 0 is the first list in `ArgsProduct`, index 1 is the second, and so on. Values are always `int64_t` regardless of what was passed (booleans like `true`/`false` become `1`/`0`).

- **Counter (`state.counters`)**
  - A named floating-point value attached to an output row. In FIDESlib, used to echo `p_batch` (batch size), `p_limbs` (limb/level count), and `p_ntt` (NTT flag) as readable columns. By default counters are shown as-is; they can also be configured as rates or thread-scaled values.

---

### FIDESlib-Specific Terms (used in benchmark context)

- **Limb**
  - A single RNS prime slot: a `VectorGPU<T>` of N elements representing one polynomial modulo one prime `p`. The fundamental unit of GPU computation. `Limb<uint32_t>` uses 32-bit coefficients; `Limb<uint64_t>` uses 64-bit.

- **RNSPoly**
  - A collection of Limbs representing one full CKKS polynomial in Residue Number System form. An RNSPoly at level `l` has `l` Limbs, one per prime. Operations on RNSPoly dispatch parallel CUDA kernels across all Limbs.

- **Level (RNS level)**
  - How many primes are currently active in an RNSPoly or ciphertext. Level 0 = top (all `L` primes active, maximum precision, most memory, most compute). Each rescale drops one prime and reduces the level by 1. `state.range(3)` in most GeneralFixture benchmarks selects which level to benchmark at.

- **ModUp**
  - Extends an RNSPoly from `L` regular limbs to `L + S` limbs by adding `S` special prime slots. Required before key switching. Expensive: O(L × S × N) operations.

- **ModDown**
  - Projects from `L + S` limbs back to `L` limbs using CRT. The inverse of ModUp. Together, ModUp + ModDown form the core of key switching (rotation and multiplication).

- **Rescale**
  - Drops the top RNS limb and adjusts remaining coefficients to divide out the dropped prime's contribution. The noise management operation after multiplication. Cheaper than ModUp/ModDown — O(L × N).

- **Key switching (KSK)**
  - The operation that transforms a degree-2 ciphertext (after multiplication) back to degree-1, using a precomputed KeySwitchingKey. Internally performs ModUp + dot product with KSK + ModDown.

- **Batch size (`p_batch`)**
  - Number of independent ciphertext operations processed together in a pipeline. Controlled by `fideslibParams.batch` and `state.range(2)`. Higher batch size amortizes fixed overheads but increases memory pressure.

- **dnum**
  - Number of "large digit" groups for hybrid key switching. Controls how many sub-polynomials a ciphertext is decomposed into during ModUp. Higher dnum = more accurate noise management but more key material and more ModUp/ModDown passes.

- **NTT (Number Theoretic Transform)**
  - The modular equivalent of the FFT. Converts a polynomial from coefficient form to evaluation form (NTT domain). Pointwise multiplication in NTT domain = polynomial multiplication in coefficient domain. FIDESlib operates mostly in NTT domain to keep ciphertexts ready for multiplication.

- **INTT (Inverse NTT)**
  - Converts back from NTT (evaluation) domain to coefficient domain. Required before rescale and certain key-switching operations.

---

## 1. Framework Overview

The benchmark suite uses **Google Benchmark** with two custom fixture classes defined in `bench/Benchmark.cuh`. Every benchmark is registered via `BENCHMARK_DEFINE_F` / `BENCHMARK_REGISTER_F`. The binary entry point is in `bench/Benchmark.cu`.

### Timing modes

| Mode | How it works | When used |
|---|---|---|
| **Automatic** (default) | Google Benchmark controls timing; wraps the entire `for (auto _ : state)` body | Operations that cannot be meaningfully reset between iterations |
| **Manual** (`UseManualTime()`) | Code calls `state.SetIterationTime(elapsed.count())` explicitly; only the bracketed region counts | Operations where setup/teardown inside the loop must be excluded |

### Global macros (`bench/Benchmark.cuh`)

| Macro | Value | Meaning |
|---|---|---|
| `PARAMETERS` | `{3, 4, 5, 6}` | Indices into `general_bench_params[]` (see §2) |
| `BATCH_CONFIG` | `{100}` | Default batch size for ciphertext-level benchmarks |
| `LEVEL_CONFIG` | `{0, 1, …, 30}` | RNS levels (number of active limbs relative to top) to sweep |
| `SYNC` | `false` | If `true`, inserts a `cudaDeviceSynchronize` every iteration; normally off for throughput |

---

## 2. Parameter Sets

### `general_bench_params[]` — 28 entries indexed by `state.range(0)`

Used by `GeneralFixture` benchmarks. Selected by `state.range(0)`.

| Index | Name | ringDim | multDepth | scaleModSize (bits) | dnum | Scaling technique |
|---|---|---|---|---|---|---|
| 0 | gparams64_13 | 2^13 = 8192 | 25 | 36 | 2 | FIXEDMANUAL |
| 1 | gparams64_14 | 2^14 = 16384 | 7 | 38 | 3 | FIXEDMANUAL |
| 2 | gparams64_15 | 2^15 = 32768 | 9 | 41 | 3 | FIXEDMANUAL |
| 3 | gparams64_16 | 2^16 = 65536 | 29 | 59 | 4 | FIXEDMANUAL |
| 4 | gparams64_16_boot2 | 2^16 | 29 | 59 | 6 | FIXEDMANUAL |
| 5 | gparams64_16_boot1 | 2^16 | 23 | 55 | 4 | FIXEDMANUAL |
| 6 | gparams64_17 | 2^17 = 131072 | 23 | 55 | 4 | FIXEDMANUAL |
| 7–13 | same ring dims | same as 0–6 | same | same | same | **FIXEDAUTO** |
| 14–20 | same ring dims | same as 0–6 | same | same | same | **FLEXIBLEAUTO** |
| 21–27 | same ring dims | same as 0–6 | same | same | same | **FLEXIBLEAUTOEXT** |

**`PARAMETERS = {3, 4, 5, 6}`** maps to indices 3–6 in this array, i.e., the four FIXEDMANUAL sets for ringDim 2^16 (large) to 2^16+bootstrap+2^17.

**Field meanings:**

- **ringDim (`N`)**: Size of the polynomial ring. Determines ciphertext size = N × (number of limbs) × 8 bytes. Larger N = more slots = more memory = more parallelism.
- **multDepth**: Maximum number of sequential multiplications. Determines how many RNS limbs (primes) the context allocates (`L = multDepth`).
- **scaleModSize**: Bit-size of each RNS prime. Trades precision vs. noise: larger bits = more precision per limb but reduces how many primes fit.
- **dnum**: Number of "large digits" for key switching (hybrid decomposition). Higher dnum = more key-switching keys but better noise growth control. Also controls how limbs are grouped for ModUp/ModDown.
- **Scaling technique**: How the scaling factor is managed across multiplications:
  - `FIXEDMANUAL` — user controls rescaling manually
  - `FIXEDAUTO` — automatic rescaling at fixed intervals
  - `FLEXIBLEAUTO` — adaptive automatic rescaling
  - `FLEXIBLEAUTOEXT` — extended adaptive rescaling (extra limb)

### `fideslib_bench_params[]` — 9 entries indexed by `state.range(1)` (FIDESlibFixture only)

These are the internal FIDESlib `Parameters` structs, separate from the OpenFHE-facing `GeneralBenchParams`.

| Index | logN | L (limbs) | dnum |
|---|---|---|---|
| 0 | 15 | 32 | 4 |
| 1 | 16 | 32 | 4 |
| 2 | 13 | 25 | 2 |
| 3 | 16 | 29 | 4 |
| 4 | 14 | 7 | 3 |
| 5 | 15 | 9 | 3 |
| 6 | 13 | 32 | 4 |
| 7 | 14 | 32 | 4 |
| 8 | 15 | 32 | 4 |

---

## 3. Fixture Classes

### `GeneralFixture` (`bench/Benchmark.cuh:81`)

High-level fixture. Used for all ciphertext-level operations.

**Setup (per benchmark invocation):**
1. Reads `state.range(0)` → selects from `general_bench_params[]`
2. Reads `state.range(1)` → selects from `fideslib_bench_params[]` (always `{0}` in most registrations)
3. Creates and **caches** an OpenFHE `CryptoContext` keyed by `state.range(0)` — expensive contexts are reused across benchmark iterations
4. Generates keys (PKE, KeySwitch, EvalMult, EvalRotate for indices {1,2,3,4})
5. All features enabled: PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE

**Available in every GeneralFixture benchmark:**
- `cc` — OpenFHE crypto context
- `keys` — key pair (publicKey, secretKey, evalKey)
- `generalTestParams` — the selected `GeneralBenchParams`
- `fideslibParams` — the selected FIDESlib `Parameters`

### `FIDESlibFixture` (`bench/Benchmark.cuh:149`)

Low-level fixture. Used for Limb and RNSPoly operations that bypass OpenFHE entirely.

**Setup:** Only reads `state.range(0)` → selects from `fideslib_bench_params[]`. No OpenFHE context, no key generation.

---

## 4. `state.range(N)` — Parameter Slot Reference

This is the most important thing to understand. Each benchmark registers with `->ArgsProduct({list0, list1, list2, list3, ...})`. Google Benchmark creates one run for every combination. `state.range(N)` reads the Nth value for a given run.

### GeneralFixture benchmarks (most common pattern)

```
ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})
            range(0)     range(1) range(2)   range(3)
```

| `state.range(N)` | Alias | Meaning |
|---|---|---|
| `state.range(0)` | param set index | Index into `general_bench_params[]`. Selects ringDim, multDepth, scaleModSize, dnum, scaling technique. |
| `state.range(1)` | FIDESlib param index | Index into `fideslib_bench_params[]`. Almost always `0` in GeneralFixture registrations. |
| `state.range(2)` | batch size | Number of independent ciphertexts processed together (pipeline/batch). Stored in `fideslibParams.batch`. Default: 100. |
| `state.range(3)` | RNS level | Which level in the modular chain to operate at. Level 0 = top (all limbs active = most expensive). Level L-1 = lowest (fewest limbs). Benchmarks skip if `multDepth <= level`. |

### FIDESlibFixture benchmarks

```
ArgsProduct({{1}, {0, 1, 8, 16}})
            range(0)  range(1)
```

| `state.range(N)` | Alias | Meaning |
|---|---|---|
| `state.range(0)` | FIDESlib param index | Index into `fideslib_bench_params[]`. Usually `{1}` or `{2}` for these lower-level benches. |
| `state.range(1)` | limb count | Number of RNS limbs (primes) to include in the operation. Controls the polynomial size indirectly. |
| `state.range(2)` | limb level (RNSPoly benches) | Starting level for the RNSPoly, same concept as level in GeneralFixture. |

### Special extra parameters

Some benchmarks add a 5th parameter (`state.range(4)`):

| Benchmark | `state.range(4)` | Meaning |
|---|---|---|
| `CiphertextRotateAndAccumulate` | `{2, 4, 8}` | Baby-step size `bstep` for the tree-accumulation rotation pattern |
| `RNSPolyStandardModDown` / variants | `state.range(1)` = `{true, false}` | Whether to apply NTT before moddown (`p_ntt` counter) |

---

## 5. Custom Counters (`state.counters[...]`)

Counters appear as extra columns in benchmark output. They are for human readability — they do not affect timing.

| Counter | Set by | Meaning |
|---|---|---|
| `p_batch` | `state.counters["p_batch"] = state.range(2)` | The batch size used in this run. Echoes `state.range(2)`. |
| `p_limbs` | `state.counters["p_limbs"] = state.range(3)` or `state.range(1)` | The RNS level or limb count used. Echoes the relevant range parameter. |
| `p_ntt` | `state.counters["p_ntt"] = state.range(1)` | Whether NTT was applied before the moddown (ModDown benchmarks only). |

---

## 6. Benchmark Catalogue

### 6.1 `ContextBenchmarks.cu` — `FIDESlibFixture`

#### `ContextCreation`
- **What it does:** Times `GenCryptoContextGPU()` — allocating and initializing the GPU crypto context, including uploading prime tables and precomputations to GPU memory.
- **Registration:** `ArgsProduct({PARAMETERS})` — 4 runs, one per parameter set index.
- **Timing:** Automatic (entire loop body timed).
- **Parameters:** `state.range(0)` = FIDESlib param index.
- **Use:** Baseline cost for context initialization. Not a per-operation measurement.

---

### 6.2 `LimbBatchAddSubBenchmarks.cu` — `FIDESlibFixture`

Operates at the lowest abstraction level: individual `Limb<T>` objects on one GPU.

A **Limb** is a single RNS prime slot: a `VectorGPU<T>` of length N holding one polynomial modulo one prime `p`.

#### `LimbBatchAdd` / `LimbBatchAdd64`
- **What it does:** Creates `n` pairs of 32-bit (or 64-bit) Limbs, then times running `limb.add(other)` sequentially across all `n` pairs. Each add is a CUDA kernel: `a[i] += b[i] mod p` for all N polynomial coefficients.
- **Parameters:**
  - `state.range(0)` = FIDESlib param index (always `{1}`)
  - `state.range(1)` = `n` = number of limb pairs = `p_limbs` counter. Values: `{1, 8, 16, 32, 64, 128}`.
- **Timing:** Manual — excludes limb allocation and data load.
- **What to observe:** How wall-clock time scales with `n`. Reveals whether multiple kernel launches (one per limb) overlap on the GPU or serialize.

#### `LimbBatchSub` / `LimbBatchSub64`
- **What it does:** Same as LimbBatchAdd but subtraction (`a[i] -= b[i] mod p`).
- **Parameters:** Identical to LimbBatchAdd.

---

### 6.3 `LimbNTTBenchmarks.cu` — `FIDESlibFixture`

Benchmarks the Number Theoretic Transform (NTT) and its inverse (INTT) on single Limbs and batched RNSPolys, across multiple reduction algorithms.

An **NTT** converts a polynomial from coefficient domain to evaluation domain (needed before pointwise multiplication). An **INTT** is the reverse.

#### Single-limb NTT/INTT — 32-bit and 64-bit

Pattern: `Limb<uint32_t or uint64_t>`, one GPU, single prime, N coefficients.

| Benchmark | Algorithm | Template |
|---|---|---|
| `LimbINTT32` / `LimbINTT64` | ALGO_NATIVE — standard modular arithmetic | `limb.INTT<ALGO_NATIVE>()` |
| `LimbNothingINTT32` / `LimbNothingNTT32` | ALGO_NONE — butterfly without reduction | `limb.INTT<ALGO_NONE>()` |
| `LimbShoupINTT32` / `LimbShoupNTT32` | ALGO_SHOUP — Shoup precomputed constants | `limb.INTT<ALGO_SHOUP>()` |
| `LimbBarretINTT32` / `LimbBarretNTT32` | ALGO_BARRETT — Barrett reduction | `limb.INTT<ALGO_BARRETT>()` |
| `LimbFP64Accel53bitINTT32` / `NTT32` | ALGO_BARRETT_FP64 — Barrett with 53-bit FP64 | `limb.INTT<ALGO_BARRETT_FP64>()` |

- **Parameters:** `state.range(0)` = FIDESlib param index (`{0}` for 32-bit, `{1}` for 64-bit).
- **Timing:** Automatic (Google Benchmark times the INTT/NTT call directly). No `UseManualTime`.
- **What to observe:** Compare algorithms. ALGO_SHOUP is typically fastest because it replaces a modular reduction with a multiply-shift using a precomputed constant.

#### Batched device NTT/INTT

`LimbDeviceBatchINTT64`, `LimbDeviceBatchNTT64`, `LimbDeviceBatchINTT32`, `LimbDeviceBatchNTT32`

- **What it does:** Creates an RNSPoly with `n` limbs and times INTT/NTT on the entire polynomial (all limbs in parallel on the GPU via streams).
- **Parameters:**
  - `state.range(0)` = FIDESlib param index
  - `state.range(1)` = number of limbs (`{1, 8, 16, 32, 64, 128}`)
- **Timing:** Manual.
- **What to observe:** Scalability — how throughput changes as more limbs are processed concurrently.

---

### 6.4 `RNSPolyAdditionBenchmarks.cu` — `FIDESlibFixture`

Operates at **RNSPoly** level: a collection of Limbs representing a full polynomial in RNS form.

#### `RNSPolyAdd`
- **What it does:** Creates two RNSPolys with `state.range(1)` limbs, times `a.add(b)` — element-wise modular addition across all limbs in parallel.
- **Parameters:**
  - `state.range(0)` = FIDESlib param index (`{1}`)
  - `state.range(1)` = limb count = `p_limbs`. Values: `{0, 1, 8, 16}`.
- **Timing:** Manual.
- **Note:** Creates new RNSPoly objects inside the timing loop — includes allocation cost.

#### `RNSPolyAddContextLimbCount`
- **What it does:** Same as RNSPolyAdd but uses `cc->L` (maximum limbs from context) instead of a fixed count. Measures the realistic full-size cost.
- **Parameters:** `state.range(0)` only.

#### `RNSPolyMultiAdd`
- **What it does:** Times 10 sequential `a.add(b)` calls on the same polynomials. Amortizes kernel launch overhead over 10 operations; time reported is total for 10 adds.
- **Parameters:** Same as RNSPolyAdd.

#### `RNSPolyMultiAddContextLimbCount`
- **What it does:** 10 sequential adds at full context limb count.

#### `RNSPolySub` / `RNSPolySubContextLimbCount`
- **What it does:** Same as Add variants but subtraction (`a.sub(b)`).

---

### 6.5 `RNSPolyRescaleBenchmarks.cu` — `FIDESlibFixture`

#### `RNSPolyRescale`
- **What it does:** Times `a.rescale()` — drops the top RNS limb and adjusts remaining coefficients to account for the removed prime. This is the noise-management operation performed after multiplication.
- **Parameters:**
  - `state.range(0)` = FIDESlib param index
  - `state.range(1)` = starting limb count = `p_limbs`. Values: `{0, 1, …, 30}` (LEVEL_CONFIG).
  - `state.range(2)` = batch size = `p_batch`.
- **Timing:** Manual — creates a new RNSPoly each iteration (includes allocation).
- **Note:** Skips if `cc->L <= state.range(1)` (not enough limbs for that level).

#### `RNSPolyRescaleContextLimbCount`
- **What it does:** Times rescale at `cc->L - state.range(3)` limbs. After rescaling, calls `a.grow(cc->L)` to restore level for the next iteration.
- **Parameters:** `state.range(0)`, `state.range(2)` (batch), `state.range(3)` (level offset).
- **Timing:** Manual — the `grow()` call is outside the timed region.
- **What to observe:** How rescale cost varies with initial limb count (proportional to N × limb_count).

---

### 6.6 `RNSPolyModUpModDownBenchmarks.cu` — `FIDESlibFixture`

ModUp extends an RNSPoly from `L` limbs to `L + S` limbs (adding the special prime slots). ModDown reduces back. These are the core of key switching.

`state.range(1)` = NTT flag (`{true, false}`) — whether to apply NTT as part of moddown. Stored as `p_ntt` counter.

#### `RNSPolyModUp` / `RNSPolyModUpContextLimbCount`
- **What it does:** Times `a.modup()` alone. Setup: creates RNSPoly at `state.range(2)` limbs.
- **Parameters:** `{2}`, `{true, false}`, `{0, 1, 8, 16}` for state.range(0/1/2).
- **Timing:** Manual.

#### `RNSPolyStandardModDown` / `ContextLimbCount`
- **What it does:** Times `a.moddown<ALGO_NATIVE>(ntt_flag)` only. Setup runs `modup()` + `generateSpecialLimbs()` first (outside timed region).
- **Algorithm:** ALGO_NATIVE (standard C++ modular arithmetic).

#### `RNSPolyStandardModUpModDown` / `ContextLimbCount`
- **What it does:** Times the **full sequence**: `modup() + generateSpecialLimbs() + moddown()` all in one timed block. This is the "combined" benchmark for comparison with separate ModUp + ModDown.

#### `RNSPolyNoneModDown` / `NoneModUpModDown` variants
- Same pattern but `moddown<ALGO_NONE>` — butterfly without modular reduction. Used to measure the cost of the butterfly structure alone, isolating reduction overhead.

#### `RNSPolyShoupNoRedModDown` / `ShoupNoRedModUpModDown` variants
- `moddown<ALGO_SHOUP>` — Shoup precomputed reduction. Typically faster than NATIVE.

#### `RNSPolyBarretModDown` / `BarretModUpModDown` variants
- `moddown<ALGO_BARRETT>` — Barrett reduction.
- **Extra parameters:** `{2, 3, 4, 5}` for `state.range(0)` (more param sets tested for Barrett).

---

### 6.7 `CiphertextAddBenchmarks.cu` — `GeneralFixture`

All use automatic timing (no `UseManualTime`). Error checking every 100 iterations (`SYNC=false`).

#### `CiphertextAdd`
- **What it does:** Times `ct1.add(ct2)` — ciphertext + ciphertext addition. Internally: element-wise add across both polynomial components (c0, c1), across all active RNS limbs.
- **Parameters:** `PARAMETERS × {0} × BATCH_CONFIG × LEVEL_CONFIG`
  - `state.range(3)` = level → determines how many limbs are active.
- **Complexity:** O(N × level) modular additions, embarrassingly parallel on GPU.

#### `AddPlaintext`
- **What it does:** Times `ct1.addPt(pt2)` — adds an (unencrypted) plaintext polynomial to a ciphertext. Only c0 is modified; c1 is unchanged.
- **Parameters:** Same as CiphertextAdd.

#### `AddScalar`
- **What it does:** Times `ct1.addScalar(1.00123123)` — adds a constant scalar to all slots. The scalar is broadcast-encoded and added to c0 only.
- **Parameters:** Same as CiphertextAdd.

---

### 6.8 `CiphertextMultiplicationBenchmarks.cu` — `GeneralFixture`

All use automatic timing.

#### `CiphertextMultiplication`
- **What it does:** Times `ct1.mult(ct2, false)` — full ciphertext × ciphertext multiplication. After the call, resets `NoiseFactor` and `NoiseLevel` to simulate a fresh multiplication for the next iteration.
- **What happens internally:**
  1. Tensor product: (c0, c1) × (d0, d1) → (e0, e1, e2) — 4 RNS polynomial multiplications
  2. Key switching of e2: ModUp → dot product with eval key → ModDown → accumulate into (e0, e1)
  3. Result is a fresh degree-1 ciphertext with doubled noise level
- **Parameters:** `PARAMETERS × {0} × BATCH_CONFIG × LEVEL_CONFIG`. Skips if `multDepth <= level`.
- **Note:** Requires eval multiplication key, loaded via `GPUcc->AddEvalKey(kskEval)`.

#### `CiphertextSquaring`
- **What it does:** Times `ct1.square(false)` — optimized self-multiplication. Skips the symmetric cross-products, doing 3 multiplications instead of 4 (uses `binomialSquare` kernel).
- **Parameters:** Same as CiphertextMultiplication.

#### `MultScalar`
- **What it does:** Times `ct1.multScalar(1.01231331, false)` — multiply all slots by a constant. Encodes the scalar into an RNS representation and does N × level pointwise multiplications. No key switching needed.
- **Parameters:** Same. Faster than full multiplication because no key switching.

---

### 6.9 `CiphertextRotationBenchmarks.cu` — `GeneralFixture`

All use automatic timing.

#### `CiphertextRotation`
- **What it does:** Times `ct1.rotate(1, true)` — cyclic shift of slots by 1 position. Internally applies the Galois automorphism (index permutation of polynomial coefficients) followed by key switching.
- **Parameters:** `PARAMETERS × {0} × BATCH_CONFIG × LEVEL_CONFIG`. Generates rotation key for index 1.
- **Note:** `true` = apply NTT before the dot product (standard mode).

#### `CiphertextHoistedRotation`
- **What it does:** Times `ct1.rotate_hoisted({1,2,3,4}, {&ct2,&ct3,&ct4,&ct1}, false)` — four rotations sharing a single ModUp computation ("hoisting"). More efficient than 4 separate rotations when the same ciphertext is rotated multiple times.
- **What hoisting means:** ModUp (the expensive part of key switching) is computed once; the dot product against 4 different rotation keys is computed in parallel.
- **Parameters:** Same. Requires rotation keys for {1,2,3,4}.
- **What to observe:** Compare with 4× `CiphertextRotation` time. Speedup shows the benefit of hoisted key switching.

#### `CiphertextRotateAndAccumulate`
- **What it does:** Times `Accumulate(ct1, bstep, 1, N/2)` — a tree-reduction that sums all slots by performing ⌈log_bstep(N/2)⌉ rounds of rotate-then-add.
- **Parameters:** Adds `state.range(4)` = `{2, 4, 8}` = baby-step size `bstep`. Larger bstep = fewer rounds but more rotations per round.
- **What to observe:** Total cost of a full inner product reduction, used in matrix-vector multiplication and bootstrapping.

---

### 6.10 `CiphertextModUpModDownBenchmarks.cu` — `GeneralFixture`

All use `UseManualTime`. No `LEVEL_CONFIG` — these do not vary by level (uses default top-level).

#### `CiphertextModUp`
- **What it does:** Times `ct1.modUp()` alone. After timing, runs `generateSpecialLimbs() + modDown()` to restore the ciphertext for the next iteration (both outside timed region).
- **What ModUp does:** Extends ciphertext from L limbs to L+S limbs by lifting coefficients into the special prime basis. Required before key switching.

#### `CiphertextModDown`
- **What it does:** Times `ct1.modDown()` alone. Setup (outside timed region): `modUp() + generateSpecialLimbs()`.
- **What ModDown does:** Projects from L+S limbs back to L limbs using the Chinese Remainder Theorem, incorporating the key-switching correction.

#### `CiphertextModUpModDown`
- **What it does:** Times the **full sequence** `modUp() + generateSpecialLimbs() + modDown()` in one block (automatic timing).
- **Use:** Compare `time(ModUp) + time(ModDown)` vs `time(ModUpModDown combined)` to quantify kernel-launch overhead.

---

### 6.11 `PlaintextMultiplicationBenchmarks.cu` — `GeneralFixture`

Uses a custom timing loop (warm-up detection pattern), not `UseManualTime`.

#### `MultPlaintext`
- **What it does:** Times `ct1.multPt(pt2, false)` — multiply ciphertext by plaintext polynomial. Applies INTT to plaintext, pointwise multiply in coefficient domain, no key switching. Resets `NoiseLevel = 1` each iteration.
- **Timing:** Custom warm-up loop: runs 1, 10, 100, … iterations until total time exceeds 1 second, then prints microseconds/iteration. (The Google Benchmark `for (auto _ : state)` body is currently inside `if constexpr (0)` — disabled — so the custom loop runs instead but does not feed timing back to Google Benchmark.)

#### `Rescale`
- **What it does:** Times `ct1.rescale()` — drops the top limb and adjusts the scale factor. After each rescale, calls `grow()` and resets `NoiseLevel = 2` to restore the ciphertext.
- **Timing:** Same custom warm-up loop.
- **Note:** The ciphertext is encoded at `NoiseLevel=2` (as if after a multiplication) so rescale has the correct precondition.

#### `AdjustAddSub`, `AdjustMult`, `AdjustPlaintext`
- **Currently disabled** (commented out in `BENCHMARK_REGISTER_F`).
- **What they would do:** Benchmark the level-adjustment operations that align two ciphertexts/plaintexts to the same RNS level before an arithmetic operation.

---

### 6.12 `MatrixVectorMultiplicationBenchmarks.cu` — `GeneralFixture`

Application-level benchmarks demonstrating real workloads.

#### `GPUMatVecMult`
- **What it does:** Times an 8-row matrix × vector multiplication using the "diagonal encoding" approach:
  1. `ct[0].multPt(pt[0])` — first row
  2. For rows 1–7: `ct[0].addMultPt(ct[i], pt[i])` — fused multiply-add (uses `mult1Add2_` kernel)
  3. `ct[0].rescale()` — noise reduction
  4. `ct[0].grow(L)` — restore level for next iteration
- **Parameters:** `PARAMETERS × {0} × BATCH_CONFIG × {0}` (always at level 0).
- **What to observe:** Full pipeline cost for a single matrix row evaluation. Exercises multPt, addMultPt (already fused), and rescale together.

#### `GPUMatVecMultScalar`
- **What it does:** Same matrix-vector multiplication but using scalar multiplications (no plaintexts). Each row multiplied by a double scalar.

#### `CPUMatVecMult` / `CPUMatVecMultScalar`
- **What they do:** CPU reference implementations for comparison. Not GPU-accelerated.

#### `GPUMatVecMultWSum`
- **What it does:** Matrix-vector multiplication followed by a weighted sum reduction.

---

### 6.13 `UnfusedCandidateBenchmarks.cu` — `GeneralFixture` (new)

Benchmarks for identifying kernel fusion opportunities. Each unfused pair is measured individually and combined to quantify overhead.

All use `UseManualTime` with `cudaDeviceSynchronize` for precise GPU-wall-clock measurement.

**Parameters for all:** `PARAMETERS × {0} × BATCH_CONFIG × LEVEL_CONFIG`

#### Pair 1: MultPt vs. Rescale vs. MultPt+Rescale

| Benchmark | Timed region | After timing (restore state) |
|---|---|---|
| `MultPt_Alone` | `ct.multPt(pt, false)` | `NoiseLevel = 1` |
| `Rescale_Alone` | `ct.rescale()` | `grow(level+1); NoiseLevel = 2` |
| `MultPtRescale_Combined` | `ct.multPt(pt, false)` + `ct.rescale()` | `grow(level+1); NoiseLevel = 1` |

- `Rescale_Alone` encodes plaintext at `NoiseLevel=2` (post-multiplication state) so rescale is valid.
- `MultPtRescale_Combined` skips if `multDepth <= level+1` (needs one extra limb for rescale).

#### Pair 2: Ct×Ct Mult vs. Mult+Rescale

| Benchmark | Timed region | After timing |
|---|---|---|
| `Mult_Alone` | `ct1.mult(ct2, false)` | `NoiseFactor = ct2.NoiseFactor; NoiseLevel = 1` |
| `MultRescale_Combined` | `ct1.mult(ct2, false)` + `ct1.rescale()` | `grow(level+1); NoiseFactor reset; NoiseLevel = 1` |

- Requires eval multiplication key (loaded via `GPUcc->AddEvalKey`).

#### Pair 3: Add vs. Add+Rescale

| Benchmark | Timed region | After timing |
|---|---|---|
| `Add_Alone` | `ct1.add(ct2)` | (no restore needed — add is idempotent for timing) |
| `AddRescale_Combined` | `ct1.add(ct2)` + `ct1.rescale()` | `grow(level+1); NoiseLevel = 2` |

- Both ciphertexts encoded at `NoiseLevel=2` for AddRescale (rescale precondition).

**How to interpret results:**

If `time(X_Alone) + time(Rescale_Alone) > time(XRescale_Combined)` by more than ~5%, kernel-launch overhead and memory round-trips between kernels are significant — a fused kernel is worth implementing.

---

## 7. Benchmark Name Format

Google Benchmark names fixture benchmarks as:

```
FixtureName/BenchmarkName/range0/range1/range2/range3[/range4]/manual_time
```

**Example:** `GeneralFixture/MultPtRescale_Combined/3/0/100/0/manual_time`

| Segment | Value | Meaning |
|---|---|---|
| `GeneralFixture` | fixture | High-level crypto context fixture |
| `MultPtRescale_Combined` | benchmark | The specific benchmark function |
| `3` | range(0) = 3 | `general_bench_params[3]` → gparams64_16 (ringDim=2^16, multDepth=29) |
| `0` | range(1) = 0 | FIDESlib param index 0 |
| `100` | range(2) = 100 | Batch size = 100 |
| `0` | range(3) = 0 | RNS level 0 (top level, all limbs active — most expensive) |
| `manual_time` | timing mode | `SetIterationTime()` used; GPU sync included in measurement |

---

## 8. Running Benchmarks

```bash
# Run all benchmarks
./fideslib-bench

# Run a specific benchmark by name (regex)
./fideslib-bench --benchmark_filter="GeneralFixture/CiphertextMultiplication"

# Run multiple benchmarks
./fideslib-bench --benchmark_filter="MultPt_Alone|Rescale_Alone|MultPtRescale_Combined"

# Run at a specific parameter combination: param=3, fidesparam=0, batch=100, level=0
./fideslib-bench --benchmark_filter="GeneralFixture/MultPtRescale_Combined/3/0/100/0"

# Output to JSON for analysis
./fideslib-bench --benchmark_out=results.json --benchmark_out_format=json

# Control GPU count (overrides default {0})
FIDESLIB_USE_NUM_GPUS=2 ./fideslib-bench
```
