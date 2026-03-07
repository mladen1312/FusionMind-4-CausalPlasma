// Benchmark C++ Dynamic Overseer latency
#include "dynamic_overseer.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main() {
    fusionmind::DynamicOverseer ov;
    fusionmind::OverseerConfig cfg;
    ov.configure(cfg);
    
    fusionmind::TrackResult tracks[4];
    
    // Warmup
    for (int i = 0; i < 1000; i++) {
        tracks[0] = {0.1f + (rand()%100)/1000.0f, 0.85f};
        tracks[1] = {0.1f + (rand()%100)/1000.0f, 0.80f};
        tracks[2] = {0.1f + (rand()%100)/1000.0f, 0.95f};
        tracks[3] = {0.1f + (rand()%100)/1000.0f, 0.90f};
        ov.decide(tracks);
    }
    
    // Benchmark
    const int N = 1000000;
    ov.reset();
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        tracks[0] = {0.2f + 0.001f*(i%100), 0.85f};
        tracks[1] = {0.15f + 0.001f*(i%80), 0.80f};
        tracks[2] = {0.1f + 0.002f*(i%50), 0.95f};
        tracks[3] = {0.3f + 0.003f*(i%30), 0.90f};
        volatile auto d = ov.decide(tracks);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (double)N;
    printf("Dynamic Overseer: %.1f ns per decision (%d calls)\n", ns, N);
    printf("  = %.3f μs = %.6f ms\n", ns/1000, ns/1e6);
    
    // Test with high disagreement
    ov.reset();
    tracks[0] = {0.9f, 0.85f};  // A says disruption
    tracks[1] = {0.1f, 0.80f};  // B says safe
    tracks[2] = {0.7f, 0.95f};  // C says danger
    tracks[3] = {0.8f, 0.90f};  // D says danger
    auto d = ov.decide(tracks);
    printf("\nHigh disagreement test:\n");
    printf("  final_prob=%.3f, best_track=%d, warning=%d, correction=%d\n",
           d.final_prob, d.best_track, d.warning_level, d.correction_applied);
    
    return 0;
}
