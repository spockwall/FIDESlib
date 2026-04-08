//
// Created by seyda on 9/14/24.
//

#ifndef FIDESLIB_ROTATION_CUH
#define FIDESLIB_ROTATION_CUH

#include <iostream>
#include "AddSub.cuh"
#include "ConstantsGPU.cuh"
#include "Math.cuh"

namespace FIDESlib::CKKS {

__device__ __forceinline__ uint32_t automorph_slot(const int n_bits, const int index, const uint32_t slot) {
    uint32_t j = slot;

    j = __brev(j) >> (32 - n_bits);

    uint32_t jTmp = (j << 1) + 1;
    uint32_t rotIndex = ((jTmp * index) & ((1 << (n_bits + 1)) - 1)) >> 1;

    // Bit reversal:
    rotIndex = __brev(rotIndex) >> (32 - n_bits);

    return rotIndex;
}

template <typename T>
__device__ __forceinline__ void automorph__(T* a, T* a_rot, const int n, const int n_bits, const int index,
                                            const int br) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t rotIndex = automorph_slot(n_bits, index, j);

    a_rot[rotIndex] = a[j];
}

// a_rot[rotIndex] = a[j] + b[rotIndex]   (automorph + modadd in one pass)
template <typename T>
__device__ __forceinline__ void automorphAdd__(T* a, T* a_rot, const T* b, const int n, const int n_bits,
                                               const int index, const int br, const int primeid) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t rotIndex = automorph_slot(n_bits, index, j);
    a_rot[rotIndex] = modadd(a[j], b[rotIndex], primeid);
}

// a_rot[rotIndex] = a[j] - b[rotIndex]   (automorph + modsub in one pass)
template <typename T>
__device__ __forceinline__ void automorphSub__(T* a, T* a_rot, const T* b, const int n, const int n_bits,
                                               const int index, const int br, const int primeid) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t rotIndex = automorph_slot(n_bits, index, j);
    a_rot[rotIndex] = modsub(a[j], b[rotIndex], primeid);
}

template <typename T>
__global__ void automorph_(T* a, T* a_rot, const int index, const int br);

__global__ void automorph_multi_(void** a, void** a_rot, const int k, const int br, const int primeid_init);

// Fused automorph + add:  a_rot[perm(j)] = a[j] + b[perm(j)]
__global__ void automorphAdd_multi_(void** a, void** a_rot, void** b, const int k, const int br,
                                    const int primeid_init);

// Fused automorph + sub:  a_rot[perm(j)] = a[j] - b[perm(j)]
__global__ void automorphSub_multi_(void** a, void** a_rot, void** b, const int k, const int br,
                                    const int primeid_init);

}  // namespace FIDESlib::CKKS

#endif  //FIDESLIB_ROTATION_CUH