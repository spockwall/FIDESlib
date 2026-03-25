//
// Benchmarks for unfused kernel sequence candidates.
// Measures individual operations (A, B) and the combined sequential pair (A+B)
// to quantify kernel-launch / memory-roundtrip overhead and decide whether
// a fused kernel is worth implementing.
//
// Pairs benchmarked:
//   1. MultPt alone  |  Rescale alone  |  MultPt + Rescale combined
//   2. Mult   alone  |  Rescale alone  |  Mult   + Rescale combined
//   3. Add    alone  |  Rescale alone  |  Add    + Rescale combined
//   4. ModUp  alone  |  ModDown alone  |  ModUp  + ModDown combined
//
// Compare:  time(A+B combined) vs. time(A alone) + time(B alone)
// If combined < sum by >5%, a fused kernel is worthwhile.
//

#include <benchmark/benchmark.h>
#include <chrono>

#include "Benchmark.cuh"
#include "CKKS/KeySwitchingKey.cuh"

namespace FIDESlib::Benchmarks {

// ============================================================================
//  Pair 1 – MultPt  |  Rescale  |  MultPt + Rescale
// ============================================================================

BENCHMARK_DEFINE_F(GeneralFixture, MultPt_Alone)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L <= level");
		return;
	}

	std::vector<int> GPUs				= generalTestParams.GPUs;
	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1	  = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	auto c1					  = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawPlainText rawPt = FIDESlib::CKKS::GetRawPlainText(cc, ptxt1);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	FIDESlib::CKKS::Plaintext GPUpt(GPUcc, rawPt);

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.multPt(GPUpt, false);
		cudaDeviceSynchronize();
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		// Restore noise level so the next iteration is identical
		GPUct1.NoiseLevel = 1;
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, Rescale_Alone)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3)) + 1) {
		state.SkipWithMessage("cc.L <= level+1 (need room for rescale)");
		return;
	}

	std::vector<int> GPUs				= generalTestParams.GPUs;
	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	// Encode at NoiseLevel=2 so rescale has something to consume
	std::vector<double> x1	  = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 2, state.range(3));
	auto c1					  = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.rescale();
		cudaDeviceSynchronize();
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		// Grow back so the next iteration starts at the same level
		GPUct1.c0.grow(GPUct1.c0.getLevel() + 1);
		GPUct1.c1.grow(GPUct1.c1.getLevel() + 1);
		GPUct1.NoiseLevel = 2;
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, MultPtRescale_Combined)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3)) + 1) {
		state.SkipWithMessage("cc.L <= level+1 (need room for rescale)");
		return;
	}

	std::vector<int> GPUs				= generalTestParams.GPUs;
	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1	  = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	auto c1					  = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawPlainText rawPt = FIDESlib::CKKS::GetRawPlainText(cc, ptxt1);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	FIDESlib::CKKS::Plaintext GPUpt(GPUcc, rawPt);

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.multPt(GPUpt, false);
		GPUct1.rescale();
		cudaDeviceSynchronize();
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		// Restore to original level
		GPUct1.c0.grow(GPUct1.c0.getLevel() + 1);
		GPUct1.c1.grow(GPUct1.c1.getLevel() + 1);
		GPUct1.NoiseLevel = 1;
	}
	CudaCheckErrorMod;
}

// ============================================================================
//  Pair 2 – Ct×Ct Mult  |  Rescale  |  Mult + Rescale
// ============================================================================

BENCHMARK_DEFINE_F(GeneralFixture, Mult_Alone)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L <= level");
		return;
	}

	std::vector<int> GPUs				= generalTestParams.GPUs;
	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1	  = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	ptxt1->SetLevel(state.range(3));
	ptxt2->SetLevel(state.range(3));
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);

	FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	kskEval.Initialize(rawKskEval);
	GPUcc->AddEvalKey(std::move(kskEval));

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.mult(GPUct2, false);
		cudaDeviceSynchronize();
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		GPUct1.NoiseFactor = GPUct2.NoiseFactor;
		GPUct1.NoiseLevel  = 1;
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, MultRescale_Rescale_Alone)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3)) + 1) {
		state.SkipWithMessage("cc.L <= level+1 (need room for rescale)");
		return;
	}

	std::vector<int> GPUs				= generalTestParams.GPUs;
	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	// NoiseLevel=2 simulates the state after a Ct×Ct mult (which raises noise by 1).
	std::vector<double> x1	  = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 2, state.range(3));
	auto c1					  = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.rescale();
		cudaDeviceSynchronize();
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		GPUct1.c0.grow(GPUct1.c0.getLevel() + 1);
		GPUct1.c1.grow(GPUct1.c1.getLevel() + 1);
		GPUct1.NoiseLevel = 2;
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, MultRescale_Combined)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3)) + 1) {
		state.SkipWithMessage("cc.L <= level+1 (need room for rescale)");
		return;
	}

	std::vector<int> GPUs				= generalTestParams.GPUs;
	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1	  = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	ptxt1->SetLevel(state.range(3));
	ptxt2->SetLevel(state.range(3));
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);

	FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	kskEval.Initialize(rawKskEval);
	GPUcc->AddEvalKey(std::move(kskEval));

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.mult(GPUct2, false);
		GPUct1.rescale();
		cudaDeviceSynchronize();
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		// Restore level and noise for next iteration
		GPUct1.c0.grow(GPUct1.c0.getLevel() + 1);
		GPUct1.c1.grow(GPUct1.c1.getLevel() + 1);
		GPUct1.NoiseFactor = GPUct2.NoiseFactor;
		GPUct1.NoiseLevel  = 1;
	}
	CudaCheckErrorMod;
}

// ============================================================================
//  Pair 3 – Add  |  Rescale  |  Add + Rescale
// ============================================================================

BENCHMARK_DEFINE_F(GeneralFixture, Add_Alone)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L <= level");
		return;
	}

	std::vector<int> GPUs				= generalTestParams.GPUs;
	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1	  = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	auto c1					  = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2					  = cc->Encrypt(keys.publicKey, ptxt2);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.add(GPUct2);
		cudaDeviceSynchronize();
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, AddRescale_Rescale_Alone)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3)) + 1) {
		state.SkipWithMessage("cc.L <= level+1 (need room for rescale)");
		return;
	}

	std::vector<int> GPUs				= generalTestParams.GPUs;
	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	// NoiseLevel=2 simulates the state after an add that bumped the noise level.
	std::vector<double> x1	  = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 2, state.range(3));
	auto c1					  = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.rescale();
		cudaDeviceSynchronize();
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		GPUct1.c0.grow(GPUct1.c0.getLevel() + 1);
		GPUct1.c1.grow(GPUct1.c1.getLevel() + 1);
		GPUct1.NoiseLevel = 2;
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, AddRescale_Combined)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3)) + 1) {
		state.SkipWithMessage("cc.L <= level+1 (need room for rescale)");
		return;
	}

	std::vector<int> GPUs				= generalTestParams.GPUs;
	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	// Encode at NoiseLevel=2 so the result of add can be rescaled
	std::vector<double> x1	  = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 2, state.range(3));
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 2, state.range(3));
	auto c1					  = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2					  = cc->Encrypt(keys.publicKey, ptxt2);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.add(GPUct2);
		GPUct1.rescale();
		cudaDeviceSynchronize();
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		// Restore level for next iteration
		GPUct1.c0.grow(GPUct1.c0.getLevel() + 1);
		GPUct1.c1.grow(GPUct1.c1.getLevel() + 1);
		GPUct1.NoiseLevel = 2;
	}
	CudaCheckErrorMod;
}

// ============================================================================
//  Pair 4 – ModUp  |  ModDown  |  ModUp + ModDown combined
//
//  ModUp extends the RNS basis by computing the ciphertext's representation
//  in the special primes (used during key switching). ModDown then collapses
//  back to the normal basis using the CRT correction factors.
//
//  Pattern mirrors CiphertextModUpModDownBenchmarks.cu but integrates the
//  three-way split (A alone / B alone / A+B combined) for overhead analysis.
// ============================================================================

BENCHMARK_DEFINE_F(GeneralFixture, ModUp_Alone)(benchmark::State& state) {
	std::vector<int> GPUs = generalTestParams.GPUs;

	// Match the pattern from CiphertextModUpModDownBenchmarks.cu:
	// Enable features and generate keys locally to ensure the ciphertext
	// has the DECOMP limb structure that modup requires.
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::LEVELEDSHE);
	auto localKeys = cc->KeyGen();

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1	  = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	auto c1					  = cc->Encrypt(localKeys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

	state.counters["p_batch"] = state.range(2);
	CudaCheckErrorMod;

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.modUp();
		CudaCheckErrorMod;
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		// Reset special limbs so the next iteration starts from the same state
		GPUct1.c0.generateSpecialLimbs(false, false);
		GPUct1.c1.generateSpecialLimbs(false, false);
		GPUct1.modDown();
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, ModDown_Alone)(benchmark::State& state) {
	std::vector<int> GPUs = generalTestParams.GPUs;

	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::LEVELEDSHE);
	auto localKeys = cc->KeyGen();

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1	  = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	auto c1					  = cc->Encrypt(localKeys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

	state.counters["p_batch"] = state.range(2);
	CudaCheckErrorMod;

	for (auto _ : state) {
		// Set up extended basis before timing modDown
		GPUct1.modUp();
		GPUct1.c0.generateSpecialLimbs(false, false);
		GPUct1.c1.generateSpecialLimbs(false, false);
		auto start = std::chrono::high_resolution_clock::now();
		GPUct1.modDown();
		CudaCheckErrorMod;
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, ModUpModDown_Combined)(benchmark::State& state) {
	std::vector<int> GPUs = generalTestParams.GPUs;

	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::LEVELEDSHE);
	auto localKeys = cc->KeyGen();

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1	  = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	auto c1					  = cc->Encrypt(localKeys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

	state.counters["p_batch"] = state.range(2);
	CudaCheckErrorMod;

	for (auto _ : state) {
		GPUct1.modUp();
		GPUct1.c0.generateSpecialLimbs(false, false);
		GPUct1.c1.generateSpecialLimbs(false, false);
		GPUct1.modDown();
		CudaCheckErrorMod;
	}
	CudaCheckErrorMod;
}

// ============================================================================
//  Registration
// ============================================================================

// Pair 1: MultPt | Rescale | MultPt+Rescale
BENCHMARK_REGISTER_F(GeneralFixture, MultPt_Alone)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG })->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, Rescale_Alone)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG })->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, MultPtRescale_Combined)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG })->UseManualTime();

// Pair 2: Mult | Rescale | Mult+Rescale
BENCHMARK_REGISTER_F(GeneralFixture, Mult_Alone)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG })->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, MultRescale_Rescale_Alone)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG })->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, MultRescale_Combined)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG })->UseManualTime();

// Pair 3: Add | Rescale | Add+Rescale
BENCHMARK_REGISTER_F(GeneralFixture, Add_Alone)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG })->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, AddRescale_Rescale_Alone)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG })->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, AddRescale_Combined)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG })->UseManualTime();

// Pair 4: ModUp | ModDown | ModUp+ModDown
// No LEVEL_CONFIG — matches CiphertextModUpModDownBenchmarks.cu pattern
// (modup/moddown operate on the full ciphertext level, not a swept level).
BENCHMARK_REGISTER_F(GeneralFixture, ModUp_Alone)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG })->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, ModDown_Alone)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG })->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, ModUpModDown_Combined)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG });

} // namespace FIDESlib::Benchmarks
