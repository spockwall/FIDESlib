//
// Benchmarks for NEW fused __global__ kernels that compose __forceinline__
// building blocks (modmult, modadd, modsub) into a single kernel.
//
// For each candidate, we benchmark:
//   - _Unfused: sequential separate API calls (e.g., multElement + sub)
//   - _Fused:   single fused API call (e.g., mult1Sub2)
//
// The fused kernels are defined in src/CKKS/ElemenwiseBatchKernels.cu,
// wrapped by LimbPartition/RNSPoly methods in LimbPartition.cu / RNSPoly.cpp.
//
// Candidates:
//   1. mult1Sub2    = modmult + modsub    (mirror of existing mult1Add2)
//   2. subMult      = modsub + modmult    (mirror of existing addMult)
//   3. addSub       = modadd + modsub
//   4. multAddSub   = modmult + modadd + modsub  (3-op chain)
//

#include <benchmark/benchmark.h>
#include <chrono>

#include "Benchmark.cuh"

namespace FIDESlib::Benchmarks {

// ============================================================================
//  Candidate 1: mult1Sub2  (this = this * p1 - p2)
//  Compare: multElement(p1) + sub(p2)  vs  mult1Sub2(p1, p2)
// ============================================================================

BENCHMARK_DEFINE_F(GeneralFixture, MultSub_Unfused)(benchmark::State& state) {
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

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c1));
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c2));
	FIDESlib::CKKS::Ciphertext GPUct3(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c3));

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		// Two separate operations: this = this * p1, then this = this - p2
		GPUct1.c0.multElement(GPUct2.c0);
		GPUct1.c0.sub(GPUct3.c0);
		GPUct1.c1.multElement(GPUct2.c1);
		GPUct1.c1.sub(GPUct3.c1);
		cudaDeviceSynchronize();
		auto end     = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, MultSub_Fused)(benchmark::State& state) {
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

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c1));
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c2));
	FIDESlib::CKKS::Ciphertext GPUct3(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c3));

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		// Single fused operation: this = this * p1 - p2
		GPUct1.c0.mult1Sub2(GPUct2.c0, GPUct3.c0);
		GPUct1.c1.mult1Sub2(GPUct2.c1, GPUct3.c1);
		cudaDeviceSynchronize();
		auto end     = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
	CudaCheckErrorMod;
}

// ============================================================================
//  Candidate 2: subMult  (this = (this - p1) * p2)
//  Compare: sub(p1) + multElement(p2)  vs  subMult(p1, p2)
// ============================================================================

BENCHMARK_DEFINE_F(GeneralFixture, SubMult_Unfused)(benchmark::State& state) {
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

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c1));
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c2));
	FIDESlib::CKKS::Ciphertext GPUct3(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c3));

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.c0.sub(GPUct2.c0);
		GPUct1.c0.multElement(GPUct3.c0);
		GPUct1.c1.sub(GPUct2.c1);
		GPUct1.c1.multElement(GPUct3.c1);
		cudaDeviceSynchronize();
		auto end     = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, SubMult_Fused)(benchmark::State& state) {
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

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c1));
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c2));
	FIDESlib::CKKS::Ciphertext GPUct3(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c3));

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.c0.subMult(GPUct2.c0, GPUct3.c0);
		GPUct1.c1.subMult(GPUct2.c1, GPUct3.c1);
		cudaDeviceSynchronize();
		auto end     = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
	CudaCheckErrorMod;
}

// ============================================================================
//  Candidate 3: addSub  (this = (this + p1) - p2)
//  Compare: add(p1) + sub(p2)  vs  addSub(p1, p2)
// ============================================================================

BENCHMARK_DEFINE_F(GeneralFixture, AddSub_Unfused)(benchmark::State& state) {
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

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c1));
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c2));
	FIDESlib::CKKS::Ciphertext GPUct3(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c3));

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.c0.add(GPUct2.c0);
		GPUct1.c0.sub(GPUct3.c0);
		GPUct1.c1.add(GPUct2.c1);
		GPUct1.c1.sub(GPUct3.c1);
		cudaDeviceSynchronize();
		auto end     = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, AddSub_Fused)(benchmark::State& state) {
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

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c1));
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c2));
	FIDESlib::CKKS::Ciphertext GPUct3(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c3));

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.c0.addSub(GPUct2.c0, GPUct3.c0);
		GPUct1.c1.addSub(GPUct2.c1, GPUct3.c1);
		cudaDeviceSynchronize();
		auto end     = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
	CudaCheckErrorMod;
}

// ============================================================================
//  Candidate 4: multAddSub  (this = this * p1 + p2 - p3)
//  Compare: multElement(p1) + add(p2) + sub(p3)  vs  multAddSub(p1, p2, p3)
// ============================================================================

BENCHMARK_DEFINE_F(GeneralFixture, MultAddSub_Unfused)(benchmark::State& state) {
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
	lbcrypto::Plaintext ptxt4 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);
	auto c4 = cc->Encrypt(keys.publicKey, ptxt4);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c1));
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c2));
	FIDESlib::CKKS::Ciphertext GPUct3(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c3));
	FIDESlib::CKKS::Ciphertext GPUct4(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c4));

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.c0.multElement(GPUct2.c0);
		GPUct1.c0.add(GPUct3.c0);
		GPUct1.c0.sub(GPUct4.c0);
		GPUct1.c1.multElement(GPUct2.c1);
		GPUct1.c1.add(GPUct3.c1);
		GPUct1.c1.sub(GPUct4.c1);
		cudaDeviceSynchronize();
		auto end     = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, MultAddSub_Fused)(benchmark::State& state) {
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
	lbcrypto::Plaintext ptxt4 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);
	auto c4 = cc->Encrypt(keys.publicKey, ptxt4);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c1));
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c2));
	FIDESlib::CKKS::Ciphertext GPUct3(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c3));
	FIDESlib::CKKS::Ciphertext GPUct4(GPUcc, FIDESlib::CKKS::GetRawCipherText(cc, c4));

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.c0.multAddSub(GPUct2.c0, GPUct3.c0, GPUct4.c0);
		GPUct1.c1.multAddSub(GPUct2.c1, GPUct3.c1, GPUct4.c1);
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

// Candidate 1: mult1Sub2
BENCHMARK_REGISTER_F(GeneralFixture, MultSub_Unfused)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, MultSub_Fused)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})->UseManualTime();

// Candidate 2: subMult
BENCHMARK_REGISTER_F(GeneralFixture, SubMult_Unfused)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, SubMult_Fused)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})->UseManualTime();

// Candidate 3: addSub
BENCHMARK_REGISTER_F(GeneralFixture, AddSub_Unfused)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, AddSub_Fused)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})->UseManualTime();

// Candidate 4: multAddSub
BENCHMARK_REGISTER_F(GeneralFixture, MultAddSub_Unfused)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, MultAddSub_Fused)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG})->UseManualTime();

}  // namespace FIDESlib::Benchmarks
