# FusionMind 4.0 — C++ Real-Time Kernels Build
# Requires: g++ >= 13, AVX-512 capable CPU, OpenMP

CXX      = g++
CXXFLAGS = -O3 -march=native -mavx512f -mfma -fopenmp -shared -fPIC \
           -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable

all: libfusionmind_rt.so libfusionmind_causal.so

libfusionmind_rt.so: fast_engine_api.cpp fast_engine.hpp
	$(CXX) $(CXXFLAGS) -o $@ fast_engine_api.cpp

libfusionmind_causal.so: causal_kernels.cpp
	$(CXX) $(CXXFLAGS) -o $@ causal_kernels.cpp

# For CPUs without AVX-512 (SSE4.2 fallback)
fallback: CXXFLAGS = -O3 -march=native -fopenmp -shared -fPIC -std=c++17
fallback: all

clean:
	rm -f *.so

bench: all
	cd ../../.. && python3 -m pytest tests/test_cpp_engine.py tests/test_causal_kernels.py -v -s --tb=short -k "speedup or latency"

.PHONY: all clean bench fallback
