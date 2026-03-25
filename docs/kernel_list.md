# FIDESlib CUDA Kernel List

## Unfused Kernels

### `src/NTT.cu`
| Kernel | Purpose |
|--------|---------|
| `Bit_Reverse<T>` | Bit reversal permutation |
| `NTT_1D<T>` | 1D NTT |
| `INTT_1D<T>` | 1D Inverse NTT |
| `test_kernel()` | Test utility |

### `src/ModMult.cu`
| Kernel | Purpose |
|--------|---------|
| `mult_<T,algo>(a, b, primeid)` | Element-wise modular multiply (in-place) |
| `mult_<T,algo>(a, b, c, primeid)` | Element-wise modular multiply (separate output) |
| `scalar_mult_<T,algo>(a, b, primeid, shoup_mu)` | Scalar multiplication |

### `src/AddSub.cu`
| Kernel | Purpose |
|--------|---------|
| `add_<T>(a, b, primeId)` | Element-wise addition |
| `sub_<T>(a, b, primeId)` | Element-wise subtraction |
| `add_(void**, void**, primeid_init)` | Batch addition |
| `sub_(void**, void**, primeid_init)` | Batch subtraction |
| `add_(void**, void**, void**, primeid_init)` | Ternary addition |
| `sub_(void**, void**, void**, primeid_init)` | Ternary subtraction |
| `scalar_add_(void**, uint64_t*, primeid_init)` | Scalar addition |
| `scalar_sub_(void**, uint64_t*, primeid_init)` | Scalar subtraction |

### `src/Rotation.cu`
| Kernel | Purpose |
|--------|---------|
| `automorph_<T>(a, a_rot, index, br)` | Automorphism / slot rotation |
| `automorph_multi_(void**, void**, k, br, primeid_init)` | Multi-limb rotation |

### `src/PeerUtils.cu`
| Kernel | Purpose |
|--------|---------|
| `notify_kernel(volatile uint32_t*, uint32_t)` | GPU sync flag |
| `notify_kernel_hostpin(TimelineSemaphore*, uint64_t)` | Notification via pinned host memory |
| `p2p_polling_kernel(volatile uint32_t*, uint32_t)` | GPU-side polling |
| `hostpin_polling_kernel(TimelineSemaphore*, uint64_t)` | Pinned host polling |
| `p2p_transfer_1d(float*, float*, size_t)` | 1D peer-to-peer memory transfer |

### `src/ConstantsGPU.cu`
| Kernel | Purpose |
|--------|---------|
| `printConstants()` | Debug: print GPU constants |

### `src/CKKS/Rescale.cu`
| Kernel | Purpose |
|--------|---------|
| `SwitchModulus<T>(src, o_primeid, dest_data, dest_primeid)` | Modulus switching |

### `src/CKKS/Conv.cu`
| Kernel | Purpose |
|--------|---------|
| `conv1_<T>(a, q_hat_inv, primeid)` | Convolution step 1 |

### `src/CKKS/ElemenwiseBatchKernels.cu`
| Kernel | Purpose |
|--------|---------|
| `addMult_<T>(l, l1, l2, primeid)` | Add then multiply |
| `addMult_(void**, void**, void**, primeid_init)` | Batch add-multiply |
| `Mult_(void**, void**, void**, primeid_init)` | Element-wise multiply |
| `square_(void**, void**, primeid_init)` | Element-wise square |
| `broadcastLimb0_(void**)` | Broadcast limb 0 |
| `broadcastLimb0_mgpu(void**, primeid_init, limb0)` | Multi-GPU broadcast limb 0 |
| `copy_(void**, void**)` | Limb copy |
| `copy1D_(void*, void*)` | 1D copy |
| `Scalar_mult_<algo>(void**, uint64_t*, primeid_init, shoup_mu)` | Scalar multiply |
| `eval_linear_w_sum_(n, a, bs, w, primeid_init)` | Linear weighted sum |
| `addScaleB_(void**, void**, void**, primeid_init)` | Add and scale by B |
| `scaleByP_(void**, primeid_init)` | Scale by P |
| `add_reuse_b___` | Batch add with buffer reuse |
| `add_reuse_scale_p_b___` | Batch add + scale by P |
| `sub_reuse_scale_p_b___` | Batch subtract + scale by P |
| `add_scale_p_reuse_b___` | Batch add then scale by P |
| `sub_scale_p_reuse_b___` | Batch subtract then scale by P |
| `copy_reuse_b___` | Batch copy |
| `copy_reuse_negative_b___` | Batch copy negated |
| `add_scalar_reuse_b___` | Batch scalar addition |
| `mult_scalar_reuse_b___` | Batch scalar multiplication |
| `sub_reuse_b___` | Batch subtraction |
| `mult_reuse_b___` | Batch multiplication |

### `src/CKKS/Context.cu`
| Kernel | Purpose |
|--------|---------|
| `dummy()` | Placeholder / warmup |

---

## Fused Kernels

### `src/NTT.cu` — NTT/INTT with mode fusion
| Kernel | Fusion Modes |
|--------|-------------|
| `NTT_<T, second, algo, mode>` | `RESCALE`, `MULTPT`, `MODDOWN`, `KSK_DOT`, `KSK_DOT_ACC` |
| `INTT_<T, second, algo, mode>` | `MULT_AND_SAVE`, `MULT_AND_ACC`, `ROTATE_AND_SAVE`, `SQUARE_AND_SAVE` |

### `src/CKKS/Conv.cu`
| Kernel | Purpose |
|--------|---------|
| `ModDown2<algo>(a, n, b, result)` | Modular reduction with composition |
| `DecompAndModUpConv<algo>(a, n, b, result)` | Decompose + modular upscale |

### `src/CKKS/ElemenwiseBatchKernels.cu`
| Kernel | Purpose |
|--------|---------|
| `mult1AddMult23Add4_(...)` | `l = l×l1 + l2×l3 + l4` in one pass |
| `mult1Add2_(...)` | `l = l×l1 + l2` in one pass |
| `multnomoddownend_(...)` | Multiply + moddown (no final step) |
| `binomial_square_fold_(...)` | Binomial square + fold |
| `binomialMult_(...)` | Binomial multiply |
| `binomialMultExtend_(...)` | Extended binomial multiply |
| `binomialSquare_(...)` | Binomial square |
| `binomialSquareExtend_(...)` | Extended binomial square |
| `binomialDotProdBatched___(...)` | Batched binomial dot product |
| `fusedDotKSK_2_(...)` | KSK dot product (key switching) |
| `hoistedRotateDotKSK_2_(...)` | Rotate + KSK dot product |
| `hoistedRotateDotKSKBatched___(...)` | Batched hoisted rotate + KSK dot (2 variants) |
| `dotProductPt_(...)` | Dot product with plaintext |
| `dotProductLtBatchedPt___(...)` | Batched dot product (5 variants) |

---

## Summary

| Category | Count |
|----------|-------|
| Unfused kernels | ~100 |
| Fused kernels | ~36 |
| **Total** | **~136** |

### Fusion Patterns

1. **Arithmetic fusion** — `mult1Add2`, `mult1AddMult23Add4`: combine multiply and add steps into a single pass
2. **NTT/INTT fusion** — `NTT_` / `INTT_` with mode template parameter: fuse rescale, moddown, multiply, or key-switch operations into the butterfly pass
3. **Key-switching fusion** — `fusedDotKSK_2_`, `hoistedRotateDotKSK_2_`: fuse rotation automorphism with KSK dot product
4. **Dot product fusion** — `dotProductPt_`, `dotProductLtBatchedPt___`: fuse multiply-accumulate across batches
5. **Binomial fusion** — `binomialMult_`, `binomialSquare_`, etc.: fuse multi-term polynomial arithmetic in one kernel
