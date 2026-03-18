# FIDESlib Quick Start Guide

Follow these steps to quickly build and run the FIDESlib project:

## 1. Clean the Build Directory (Optional)
If you have previously attempted to build the project and encountered errors, it is recommended to clean your build directory first:

```bash
cd /home/su/FIDESlib
rm -rf build/
```

## 2. Create the Build Directory
Create a new `build` directory and navigate into it:

```bash
mkdir -p build && cd build
```

## 3. Configure the Project with CMake
Configure CMake to download and install the required patched version of OpenFHE (`-DFIDESLIB_INSTALL_OPENFHE=ON`). To avoid issues with older/unsupported GPU architectures, tell CMake to detect your specific GPU architecture using `-DFIDESLIB_ARCH="native"`:

```bash
// clang++ 18
cmake .. \
  -DFIDESLIB_INSTALL_OPENFHE=ON \
  -DFIDESLIB_ARCH="native" \
  -DFIDESLIB_INSTALL_PREFIX=/tmp/fideslib_install \
  -DOPENFHE_INSTALL_PREFIX=/tmp/fideslib_install \
  -DCMAKE_C_COMPILER=clang-18 \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 \
  -DCUDA_PATH=/usr/local/cuda-13.2 \
  -DFIDESLIB_ARCH="80" \
  -DCMAKE_BUILD_TYPE=Release

// g++ 12
cmake .. \
  -DFIDESLIB_INSTALL_OPENFHE=ON \
  -DFIDESLIB_ARCH="native" \
  -DFIDESLIB_INSTALL_PREFIX=/tmp/fideslib_install \
  -DOPENFHE_INSTALL_PREFIX=/tmp/fideslib_install \
  -DCMAKE_C_COMPILER=gcc-12 \
  -DCMAKE_CXX_COMPILER=g++-12 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 \
  -DCUDA_PATH=/usr/local/cuda-13.2 \
  -DFIDESLIB_ARCH="80" \
  -DCMAKE_BUILD_TYPE=Release
```
*(Note: If `native` doesn't work, you can explicitly set the architecture for modern GPUs by passing `-DFIDESLIB_ARCH="80;86;89"`, leaving out `70`)*

Note: If you already have a `build/` directory, you don't need to rebuild everything:
```bash
cmake --build build -j
```
## 4. Build the Project
Compile the project utilizing all available CPU cores:

```bash
make -j$(nproc)
```

## 5. Next Steps
Once the build concludes successfully, you can verify everything is working by running the tests and benchmarks from within the `build` directory:

```bash
# Run the test suite
./fideslib-test

# Run the benchmark suite
./fideslib-bench
```

To install the library to your system (to `/tmp/fideslib_install`), you can run:
```bash
make install
```
