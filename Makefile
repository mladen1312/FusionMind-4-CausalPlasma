# FusionMind 4.0 — C++ Fast Engine Build System
# ===============================================
#
# Targets:
#   make all       — Build all shared libraries
#   make rt        — Build real-time inference engine only
#   make causal    — Build causal discovery kernels only
#   make bench     — Build and run benchmarks
#   make test      — Run all Python tests
#   make clean     — Remove build artifacts
#
# Requirements:
#   - GCC 11+ with AVX-512 support
#   - OpenMP
#   - Optional: OpenCL for GPU kernels

CXX      := g++
CXXFLAGS := -O3 -march=native -std=c++17 -Wall -Wno-unused-function -Wno-unused-variable
SIMD     := -mavx512f -mavx512dq -mfma
OMP      := -fopenmp
SHARED   := -shared -fPIC
DEFINES  := -D_GNU_SOURCE

CPP_DIR  := fusionmind4/realtime/cpp
OUT_DIR  := $(CPP_DIR)

# Targets
LIB_RT     := $(OUT_DIR)/libfusionmind_rt.so
LIB_CAUSAL := $(OUT_DIR)/libfusionmind_causal.so

.PHONY: all rt causal bench test clean info

all: rt causal
	@echo ""
	@echo "=== Build complete ==="
	@ls -la $(LIB_RT) $(LIB_CAUSAL)

rt: $(LIB_RT)

$(LIB_RT): $(CPP_DIR)/fast_engine_api.cpp $(CPP_DIR)/fast_engine.hpp
	$(CXX) $(CXXFLAGS) $(SIMD) $(SHARED) \
		-o $@ $<
	@echo "Built: $@ (real-time inference, target < 1 μs)"

causal: $(LIB_CAUSAL)

$(LIB_CAUSAL): $(CPP_DIR)/causal_kernels.cpp
	$(CXX) $(CXXFLAGS) $(SIMD) $(OMP) $(DEFINES) $(SHARED) \
		-o $@ $<
	@echo "Built: $@ (AVX-512 + OpenMP causal discovery)"

bench: all
	@echo ""
	@echo "=== Running benchmarks ==="
	python3 scripts/benchmark_cpp.py

test: all
	@echo ""
	@echo "=== Running tests ==="
	python3 -m pytest tests/test_cpp_engine.py tests/test_causal_kernels.py -v --tb=short

clean:
	rm -f $(LIB_RT) $(LIB_CAUSAL)
	@echo "Cleaned build artifacts"

info:
	@echo "Compiler: $(CXX)"
	@$(CXX) --version | head -1
	@echo "SIMD flags: $(SIMD)"
	@echo "CPU:"
	@cat /proc/cpuinfo | grep -m1 "model name" || echo "unknown"
	@echo "Cores: $$(nproc)"
	@echo "AVX-512:"
	@cat /proc/cpuinfo | grep -m1 flags | tr ' ' '\n' | grep avx512 | tr '\n' ' '
	@echo ""
	@echo "GPU:"
	@nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "None (OpenCL kernels ready for deployment)"
