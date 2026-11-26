# Source me (experiment 1 build toolchain). Requires SACQ_LAB = the lab root described in README.md.
L=${SACQ_LAB:?set SACQ_LAB to the lab root (see README)}
export CUDA_HOME=$L/toolchain
export PATH=$L/toolchain/bin:$PATH
export CC=$L/toolchain/bin/gcc CXX=$L/toolchain/bin/g++
export TORCH_CUDA_ARCH_LIST="12.0"
export UV_HTTP_TIMEOUT=120
export CPATH=$L/toolchain/targets/x86_64-linux/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$L/toolchain/targets/x86_64-linux/lib:$L/toolchain/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
