//
// Benchmarks for fused automorph+add/sub and rescale+add/sub kernels.
//
// Automorph fusion: combines the permutation (automorph__) with element-wise
// add/sub into a single kernel, eliminating one global memory round-trip.
//   - automorphAdd: a_rot[perm(j)] = a[j] + b[perm(j)]
//   - automorphSub: a_rot[perm(j)] = a[j] - b[perm(j)]
//
// Rescale+Add/Sub: measures rescale() followed by add()/sub() as a baseline
// for potential NTT-level fusion (rescale already uses NTT_RESCALE mode internally).
//

#include <benchmark/benchmark.h>
#include <chrono>

#include "Benchmark.cuh"

namespace FIDESlib::Benchmarks {

// ============================================================================
//  Candidate 9: automorphAdd  (dst[perm(j)] = src[j] + b[perm(j)])
//  Compare: automorph(src) + add(b)  vs  automorphAdd(src, b)
// ============================================================================

BENCHMARK_DEFINE_F(GeneralFixture, AutomorphAdd_Unfused)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L <= level");
		return;
	}

	std::vector<int> GPUs               = generalTestParams.GPUs;
	fideslibParams.batch                = state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc       = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::Ciphertext GPUct_dst(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c1));
	FIDESlib::CKKS::Ciphertext GPUct_src(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c2));
	FIDESlib::CKKS::Ciphertext GPUct_b(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c3));

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		// Two separate operations: automorph then add
		GPUct_dst.c0.automorph(1, 1, &GPUct_src.c0);
		GPUct_dst.c0.add(GPUct_b.c0);
		GPUct_dst.c1.automorph(1, 1, &GPUct_src.c1);
		GPUct_dst.c1.add(GPUct_b.c1);
		cudaDeviceSynchronize();
		auto end     = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, AutomorphAdd_Fused)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L <= level");
		return;
	}

	std::vector<int> GPUs               = generalTestParams.GPUs;
	fideslibParams.batch                = state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc       = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::Ciphertext GPUct_dst(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c1));
	FIDESlib::CKKS::Ciphertext GPUct_src(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c2));
	FIDESlib::CKKS::Ciphertext GPUct_b(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c3));

	// Precompute automorphism index for rotation by 1
	int k = GPUct_dst.c0.automorph_index_precomp(1);

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		// Single fused operation: automorph + add
		GPUct_dst.c0.automorphAdd(k, 1, GPUct_src.c0, GPUct_b.c0);
		GPUct_dst.c1.automorphAdd(k, 1, GPUct_src.c1, GPUct_b.c1);
		cudaDeviceSynchronize();
		auto end     = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
	CudaCheckErrorMod;
}

// ============================================================================
//  Candidate 10: automorphSub  (dst[perm(j)] = src[j] - b[perm(j)])
//  Compare: automorph(src) + sub(b)  vs  automorphSub(src, b)
// ============================================================================

BENCHMARK_DEFINE_F(GeneralFixture, AutomorphSub_Unfused)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L <= level");
		return;
	}

	std::vector<int> GPUs               = generalTestParams.GPUs;
	fideslibParams.batch                = state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc       = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::Ciphertext GPUct_dst(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c1));
	FIDESlib::CKKS::Ciphertext GPUct_src(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c2));
	FIDESlib::CKKS::Ciphertext GPUct_b(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c3));

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct_dst.c0.automorph(1, 1, &GPUct_src.c0);
		GPUct_dst.c0.sub(GPUct_b.c0);
		GPUct_dst.c1.automorph(1, 1, &GPUct_src.c1);
		GPUct_dst.c1.sub(GPUct_b.c1);
		cudaDeviceSynchronize();
		auto end     = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, AutomorphSub_Fused)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L <= level");
		return;
	}

	std::vector<int> GPUs               = generalTestParams.GPUs;
	fideslibParams.batch                = state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc       = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::Ciphertext GPUct_dst(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c1));
	FIDESlib::CKKS::Ciphertext GPUct_src(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c2));
	FIDESlib::CKKS::Ciphertext GPUct_b(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c3));

	int k = GPUct_dst.c0.automorph_index_precomp(1);

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct_dst.c0.automorphSub(k, 1, GPUct_src.c0, GPUct_b.c0);
		GPUct_dst.c1.automorphSub(k, 1, GPUct_src.c1, GPUct_b.c1);
		cudaDeviceSynchronize();
		auto end     = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
	CudaCheckErrorMod;
}

// ============================================================================
//  Candidate 11: rescaleAdd  (rescale(this) then add(b))
//  Baseline measurement — rescale uses NTT_RESCALE mode internally.
//  Future work: NTT_RESCALE_ADD mode could fuse both.
// ============================================================================

BENCHMARK_DEFINE_F(GeneralFixture, RescaleAdd_Unfused)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L <= level");
		return;
	}

	std::vector<int> GPUs               = generalTestParams.GPUs;
	fideslibParams.batch                = state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc       = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
	// Need level > 0 to have room for rescale (drops one level)
	int level = std::max((int)state.range(3), 1);
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, level);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, level);
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	// Re-encrypt each iteration since rescale consumes a level
	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c1));
		FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c2));
		// GPUct2 at level-1 to match post-rescale level
		GPUct2.c0.rescale();
		GPUct2.c1.rescale();
		cudaDeviceSynchronize();

		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.c0.rescale();
		GPUct1.c1.rescale();
		GPUct1.c0.add(GPUct2.c0);
		GPUct1.c1.add(GPUct2.c1);
		cudaDeviceSynchronize();
		auto end     = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
	CudaCheckErrorMod;
}

// ============================================================================
//  Candidate 12: rescaleSub  (rescale(this) then sub(b))
//  Baseline measurement — same as RescaleAdd but with sub.
// ============================================================================

BENCHMARK_DEFINE_F(GeneralFixture, RescaleSub_Unfused)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L <= level");
		return;
	}

	std::vector<int> GPUs               = generalTestParams.GPUs;
	fideslibParams.batch                = state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc       = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
	int level = std::max((int)state.range(3), 1);
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, level);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, level);
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c1));
		FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c2));
		GPUct2.c0.rescale();
		GPUct2.c1.rescale();
		cudaDeviceSynchronize();

		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.c0.rescale();
		GPUct1.c1.rescale();
		GPUct1.c0.sub(GPUct2.c0);
		GPUct1.c1.sub(GPUct2.c1);
		cudaDeviceSynchronize();
		auto end     = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
	CudaCheckErrorMod;
}

// ============================================================================
//  Registration
// ============================================================================

// Candidate 9: automorphAdd
BENCHMARK_REGISTER_F(GeneralFixture, AutomorphAdd_Unfused)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, AutomorphAdd_Fused)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})->UseManualTime();

// Candidate 10: automorphSub
BENCHMARK_REGISTER_F(GeneralFixture, AutomorphSub_Unfused)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, AutomorphSub_Fused)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})->UseManualTime();

// Candidate 11: rescaleAdd (baseline only — no fused kernel yet)
BENCHMARK_REGISTER_F(GeneralFixture, RescaleAdd_Unfused)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})->UseManualTime();

// Candidate 12: rescaleSub (baseline only — no fused kernel yet)
BENCHMARK_REGISTER_F(GeneralFixture, RescaleSub_Unfused)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})->UseManualTime();

}  // namespace FIDESlib::Benchmarks
