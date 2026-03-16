rm -rf build

mkdir build
cd build
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

ln -sf compile_commands.json ../compile_commands.json

make -j$(nproc) > ../log.txt 2>&1