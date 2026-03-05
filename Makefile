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
LIB_STACK  := $(OUT_DIR)/libfusionmind_stack.so

.PHONY: all rt causal stack bench test test-all clean info

all: rt causal stack
	@echo ""
	@echo "=== Build complete ==="
	@ls -la $(LIB_RT) $(LIB_CAUSAL) $(LIB_STACK)

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

stack: $(LIB_STACK)

$(LIB_STACK): $(CPP_DIR)/stack_api.cpp $(CPP_DIR)/stack_engine.hpp
	$(CXX) $(CXXFLAGS) $(SHARED) \
		-o $@ $<
	@echo "Built: $@ (4-layer stack: L0 realtime + L1 RL + L2 SCM + L3 safety)"

bench: all
	@echo ""
	@echo "=== Running benchmarks ==="
	python3 -c "from fusionmind4.realtime.stack_bindings import CppStack; \
		s=CppStack(10,3); r=s.benchmark(10000,['var_0','var_1']); \
		print('Stack P50: %(p50_ns).0fns  P95: %(p95_ns).0fns  Throughput: %(throughput_Mops).2f Mops' % r)"

test: all
	@echo ""
	@echo "=== Running tests ==="
	FM_SKIP_S3=1 python3 -m pytest tests/ -v --tb=short -q

test-all: all
	@echo ""
	@echo "=== Running ALL tests (including real data download) ==="
	python3 -m pytest tests/ -v --tb=short

clean:
	rm -f $(LIB_RT) $(LIB_CAUSAL) $(LIB_STACK)
	rm -f $(OUT_DIR)/*.so
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
