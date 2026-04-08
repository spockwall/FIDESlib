//
// Benchmarks for NEW fused __global__ kernels that compose __forceinline__
// building blocks (modmult, modadd, modsub) into a single kernel.
//
// For each candidate, we:
//   1. Define the NEW fused __global__ kernel
//   2. Benchmark the two SEPARATE kernel launches (unfused)
//   3. Benchmark the SINGLE fused kernel launch
//   4. Compare: if fused < unfused by >5%, the fused kernel is worthwhile
//
// Candidates:
//   1. mult1Sub2_    = modmult + modsub    (mirror of existing mult1Add2_)
//   2. subMult_      = modsub + modmult    (mirror of existing addMult_)
//   3. addSub_       = modadd + modsub     (no existing fused kernel)
//   4. multAddSub_   = modmult + modadd + modsub  (3-op chain)
//   5. scalarMultAdd_ = scalar modmult + modadd
//

#include "CKKS/ElemenwiseBatchKernels.cuh"
#include "AddSub.cuh"
#include "ModMult.cuh"
#include "ConstantsGPU.cuh"
#include <benchmark/benchmark.h>

#include "Benchmark.cuh"

namespace FIDESlib::Benchmarks {

// ============================================================================
//  NEW FUSED KERNELS (the actual contribution)
// ============================================================================

// l = l * l1 - l2   (modmult + modsub in one pass)
// Mirror of mult1Add2_ but with subtraction.
__global__ void mult1Sub2_(const __grid_constant__ int primeid_init,
                           void** l, void** l1, void** l2) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T aux = ((T*)l2[blockIdx.y])[idx];
        T res = modmult<algo>(((T*)l[blockIdx.y])[idx],
                              ((T*)l1[blockIdx.y])[idx], primeid);
        ((T*)l[blockIdx.y])[idx] = modsub(res, aux, primeid);
    } else {
        using T = uint32_t;
        T aux = ((T*)l2[blockIdx.y])[idx];
        T res = modmult<algo>(((T*)l[blockIdx.y])[idx],
                              ((T*)l1[blockIdx.y])[idx], primeid);
        ((T*)l[blockIdx.y])[idx] = modsub(res, aux, primeid);
    }
}

// l = (l - l1) * l2   (modsub + modmult in one pass)
// Mirror of addMult_ but with subtraction.
__global__ void subMult_(const __grid_constant__ int primeid_init,
                         void** l, void** l1, void** l2) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T tmp = modsub(((T*)l[blockIdx.y])[idx],
                       ((T*)l1[blockIdx.y])[idx], primeid);
        ((T*)l[blockIdx.y])[idx] = modmult<algo>(tmp, ((T*)l2[blockIdx.y])[idx], primeid);
    } else {
        using T = uint32_t;
        T tmp = modsub(((T*)l[blockIdx.y])[idx],
                       ((T*)l1[blockIdx.y])[idx], primeid);
        ((T*)l[blockIdx.y])[idx] = modmult<algo>(tmp, ((T*)l2[blockIdx.y])[idx], primeid);
    }
}

// l = (l + l1) - l2   (modadd + modsub in one pass)
__global__ void addSub_(const __grid_constant__ int primeid_init,
                        void** l, void** l1, void** l2) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T tmp = modadd(((T*)l[blockIdx.y])[idx],
                       ((T*)l1[blockIdx.y])[idx], primeid);
        ((T*)l[blockIdx.y])[idx] = modsub(tmp, ((T*)l2[blockIdx.y])[idx], primeid);
    } else {
        using T = uint32_t;
        T tmp = modadd(((T*)l[blockIdx.y])[idx],
                       ((T*)l1[blockIdx.y])[idx], primeid);
        ((T*)l[blockIdx.y])[idx] = modsub(tmp, ((T*)l2[blockIdx.y])[idx], primeid);
    }
}

// l = l * l1 + l2 - l3   (modmult + modadd + modsub in one pass)
__global__ void multAddSub_(const __grid_constant__ int primeid_init,
                            void** l, void** l1, void** l2, void** l3) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T res = modmult<algo>(((T*)l[blockIdx.y])[idx],
                              ((T*)l1[blockIdx.y])[idx], primeid);
        res = modadd(res, ((T*)l2[blockIdx.y])[idx], primeid);
        ((T*)l[blockIdx.y])[idx] = modsub(res, ((T*)l3[blockIdx.y])[idx], primeid);
    } else {
        using T = uint32_t;
        T res = modmult<algo>(((T*)l[blockIdx.y])[idx],
                              ((T*)l1[blockIdx.y])[idx], primeid);
        res = modadd(res, ((T*)l2[blockIdx.y])[idx], primeid);
        ((T*)l[blockIdx.y])[idx] = modsub(res, ((T*)l3[blockIdx.y])[idx], primeid);
    }
}

// l = l * scalar + l1   (scalar modmult + modadd in one pass)
template <ALGO algo>
__global__ void scalarMultAdd_(const __grid_constant__ int primeid_init,
                               void** l, const uint64_t* scalar,
                               const uint64_t* shoup_mu, void** l1) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T res = modmult<algo>(((T*)l[blockIdx.y])[idx],
                              scalar[primeid], primeid, shoup_mu[primeid]);
        ((T*)l[blockIdx.y])[idx] = modadd(res, ((T*)l1[blockIdx.y])[idx], primeid);
    } else {
        using T = uint32_t;
        T res = modmult<algo>(((T*)l[blockIdx.y])[idx],
                              (T)scalar[primeid], primeid, (T)shoup_mu[primeid]);
        ((T*)l[blockIdx.y])[idx] = modadd(res, ((T*)l1[blockIdx.y])[idx], primeid);
    }
}

// ============================================================================
//  Helper: allocate RNS polynomial data on GPU for low-level kernel benchmarks
// ============================================================================

struct RNSPolyBenchData {
    void** limbptrs;      // device array of limb pointers
    uint64_t** limbs;     // host array holding device pointers per limb
    int num_limbs;
    int N;
    int primeid_init;

    static RNSPolyBenchData alloc(int N, int num_limbs, int primeid_init) {
        RNSPolyBenchData d;
        d.N = N;
        d.num_limbs = num_limbs;
        d.primeid_init = primeid_init;

        d.limbs = new uint64_t*[num_limbs];
        for (int i = 0; i < num_limbs; i++) {
            cudaMalloc(&d.limbs[i], N * sizeof(uint64_t));
            cudaMemset(d.limbs[i], 1, N * sizeof(uint64_t));
        }
        cudaMalloc(&d.limbptrs, num_limbs * sizeof(void*));
        cudaMemcpy(d.limbptrs, d.limbs, num_limbs * sizeof(void*),
                   cudaMemcpyHostToDevice);
        return d;
    }

    void free() {
        for (int i = 0; i < num_limbs; i++) cudaFree(limbs[i]);
        cudaFree(limbptrs);
        delete[] limbs;
    }
};

// ============================================================================
//  Candidate 1: mult1Sub2_  (l = l*l1 - l2)
//  Compare: Mult_ + sub_  (two launches) vs mult1Sub2_ (one launch)
// ============================================================================

BENCHMARK_DEFINE_F(FIDESlibFixture, MultSub_Unfused)(benchmark::State& state) {
    int N = 1 << fideslibParams.logN;
    int L = fideslibParams.L;
    auto a  = RNSPolyBenchData::alloc(N, L, 0);
    auto b  = RNSPolyBenchData::alloc(N, L, 0);
    auto c  = RNSPolyBenchData::alloc(N, L, 0);
    dim3 grid{(uint32_t)N / 128, (uint32_t)L};
    CudaCheckErrorMod;

    for (auto _ : state) {
        // Two separate kernel launches
        CKKS::Mult_<<<grid, 128>>>(a.limbptrs, b.limbptrs, c.limbptrs, a.primeid_init);
        sub_<<<grid, 128>>>(a.limbptrs, c.limbptrs, a.primeid_init);
        cudaDeviceSynchronize();
    }

    a.free(); b.free(); c.free();
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, MultSub_Fused)(benchmark::State& state) {
    int N = 1 << fideslibParams.logN;
    int L = fideslibParams.L;
    auto a  = RNSPolyBenchData::alloc(N, L, 0);
    auto b  = RNSPolyBenchData::alloc(N, L, 0);
    auto c  = RNSPolyBenchData::alloc(N, L, 0);
    dim3 grid{(uint32_t)N / 128, (uint32_t)L};
    CudaCheckErrorMod;

    for (auto _ : state) {
        // Single fused kernel launch
        mult1Sub2_<<<grid, 128>>>(a.primeid_init, a.limbptrs, b.limbptrs, c.limbptrs);
        cudaDeviceSynchronize();
    }

    a.free(); b.free(); c.free();
    CudaCheckErrorMod;
}

// ============================================================================
//  Candidate 2: subMult_  (l = (l - l1) * l2)
//  Compare: sub_ + Mult_  (two launches) vs subMult_ (one launch)
// ============================================================================

BENCHMARK_DEFINE_F(FIDESlibFixture, SubMult_Unfused)(benchmark::State& state) {
    int N = 1 << fideslibParams.logN;
    int L = fideslibParams.L;
    auto a  = RNSPolyBenchData::alloc(N, L, 0);
    auto b  = RNSPolyBenchData::alloc(N, L, 0);
    auto c  = RNSPolyBenchData::alloc(N, L, 0);
    dim3 grid{(uint32_t)N / 128, (uint32_t)L};
    CudaCheckErrorMod;

    for (auto _ : state) {
        sub_<<<grid, 128>>>(a.limbptrs, b.limbptrs, a.primeid_init);
        CKKS::Mult_<<<grid, 128>>>(a.limbptrs, a.limbptrs, c.limbptrs, a.primeid_init);
        cudaDeviceSynchronize();
    }

    a.free(); b.free(); c.free();
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, SubMult_Fused)(benchmark::State& state) {
    int N = 1 << fideslibParams.logN;
    int L = fideslibParams.L;
    auto a  = RNSPolyBenchData::alloc(N, L, 0);
    auto b  = RNSPolyBenchData::alloc(N, L, 0);
    auto c  = RNSPolyBenchData::alloc(N, L, 0);
    dim3 grid{(uint32_t)N / 128, (uint32_t)L};
    CudaCheckErrorMod;

    for (auto _ : state) {
        subMult_<<<grid, 128>>>(a.primeid_init, a.limbptrs, b.limbptrs, c.limbptrs);
        cudaDeviceSynchronize();
    }

    a.free(); b.free(); c.free();
    CudaCheckErrorMod;
}

// ============================================================================
//  Candidate 3: addSub_  (l = (l + l1) - l2)
//  Compare: add_ + sub_  (two launches) vs addSub_ (one launch)
// ============================================================================

BENCHMARK_DEFINE_F(FIDESlibFixture, AddSub_Unfused)(benchmark::State& state) {
    int N = 1 << fideslibParams.logN;
    int L = fideslibParams.L;
    auto a  = RNSPolyBenchData::alloc(N, L, 0);
    auto b  = RNSPolyBenchData::alloc(N, L, 0);
    auto c  = RNSPolyBenchData::alloc(N, L, 0);
    dim3 grid{(uint32_t)N / 128, (uint32_t)L};
    CudaCheckErrorMod;

    for (auto _ : state) {
        add_<<<grid, 128>>>(a.limbptrs, b.limbptrs, a.primeid_init);
        sub_<<<grid, 128>>>(a.limbptrs, c.limbptrs, a.primeid_init);
        cudaDeviceSynchronize();
    }

    a.free(); b.free(); c.free();
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, AddSub_Fused)(benchmark::State& state) {
    int N = 1 << fideslibParams.logN;
    int L = fideslibParams.L;
    auto a  = RNSPolyBenchData::alloc(N, L, 0);
    auto b  = RNSPolyBenchData::alloc(N, L, 0);
    auto c  = RNSPolyBenchData::alloc(N, L, 0);
    dim3 grid{(uint32_t)N / 128, (uint32_t)L};
    CudaCheckErrorMod;

    for (auto _ : state) {
        addSub_<<<grid, 128>>>(a.primeid_init, a.limbptrs, b.limbptrs, c.limbptrs);
        cudaDeviceSynchronize();
    }

    a.free(); b.free(); c.free();
    CudaCheckErrorMod;
}

// ============================================================================
//  Candidate 4: multAddSub_  (l = l*l1 + l2 - l3)
//  Compare: Mult_ + add_ + sub_  (three launches) vs multAddSub_ (one launch)
// ============================================================================

BENCHMARK_DEFINE_F(FIDESlibFixture, MultAddSub_Unfused)(benchmark::State& state) {
    int N = 1 << fideslibParams.logN;
    int L = fideslibParams.L;
    auto a  = RNSPolyBenchData::alloc(N, L, 0);
    auto b  = RNSPolyBenchData::alloc(N, L, 0);
    auto c  = RNSPolyBenchData::alloc(N, L, 0);
    auto d  = RNSPolyBenchData::alloc(N, L, 0);
    dim3 grid{(uint32_t)N / 128, (uint32_t)L};
    CudaCheckErrorMod;

    for (auto _ : state) {
        CKKS::Mult_<<<grid, 128>>>(a.limbptrs, a.limbptrs, b.limbptrs, a.primeid_init);
        add_<<<grid, 128>>>(a.limbptrs, c.limbptrs, a.primeid_init);
        sub_<<<grid, 128>>>(a.limbptrs, d.limbptrs, a.primeid_init);
        cudaDeviceSynchronize();
    }

    a.free(); b.free(); c.free(); d.free();
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, MultAddSub_Fused)(benchmark::State& state) {
    int N = 1 << fideslibParams.logN;
    int L = fideslibParams.L;
    auto a  = RNSPolyBenchData::alloc(N, L, 0);
    auto b  = RNSPolyBenchData::alloc(N, L, 0);
    auto c  = RNSPolyBenchData::alloc(N, L, 0);
    auto d  = RNSPolyBenchData::alloc(N, L, 0);
    dim3 grid{(uint32_t)N / 128, (uint32_t)L};
    CudaCheckErrorMod;

    for (auto _ : state) {
        multAddSub_<<<grid, 128>>>(a.primeid_init, a.limbptrs, b.limbptrs, c.limbptrs, d.limbptrs);
        cudaDeviceSynchronize();
    }

    a.free(); b.free(); c.free(); d.free();
    CudaCheckErrorMod;
}

// ============================================================================
//  Registration
//
//  Uses FIDESlibFixture (low-level, no OpenFHE dependency).
//  state.range(0) selects the parameter set index, state.range(1) is unused.
// ============================================================================

#define FIDS_PARAMS {0, 1, 2, 3, 4, 5, 6, 7, 8}

// Candidate 1: mult1Sub2_
BENCHMARK_REGISTER_F(FIDESlibFixture, MultSub_Unfused)->ArgsProduct({FIDS_PARAMS, {0}});
BENCHMARK_REGISTER_F(FIDESlibFixture, MultSub_Fused)->ArgsProduct({FIDS_PARAMS, {0}});

// Candidate 2: subMult_
BENCHMARK_REGISTER_F(FIDESlibFixture, SubMult_Unfused)->ArgsProduct({FIDS_PARAMS, {0}});
BENCHMARK_REGISTER_F(FIDESlibFixture, SubMult_Fused)->ArgsProduct({FIDS_PARAMS, {0}});

// Candidate 3: addSub_
BENCHMARK_REGISTER_F(FIDESlibFixture, AddSub_Unfused)->ArgsProduct({FIDS_PARAMS, {0}});
BENCHMARK_REGISTER_F(FIDESlibFixture, AddSub_Fused)->ArgsProduct({FIDS_PARAMS, {0}});

// Candidate 4: multAddSub_
BENCHMARK_REGISTER_F(FIDESlibFixture, MultAddSub_Unfused)->ArgsProduct({FIDS_PARAMS, {0}});
BENCHMARK_REGISTER_F(FIDESlibFixture, MultAddSub_Fused)->ArgsProduct({FIDS_PARAMS, {0}});

}  // namespace FIDESlib::Benchmarks
