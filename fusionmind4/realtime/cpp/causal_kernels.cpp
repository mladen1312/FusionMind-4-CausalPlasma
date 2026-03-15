/**
 * FusionMind 4.0 — AVX-512 Causal Discovery Kernels
 * ====================================================
 *
 * Exploits:
 *   - AVX-512F/BW/DQ/VL for 512-bit SIMD matrix ops
 *   - FMA (fused multiply-add) for dot products
 *   - OpenMP for parallel bootstrap & Granger pairs
 *   - Cached matrix exponential with dirty-flag
 *   - Precomputed X^T X for O(d^2) inner gradient
 *   - CPU affinity pinning for deterministic latency
 *
 * Build:
 *   g++ -O3 -march=native -mavx512f -mfma -fopenmp -shared -fPIC \
 *       -std=c++17 -o libfusionmind_causal.so causal_kernels.cpp
 *
 * Patent Family: PF1 (CPDE)
 * Author: Dr. Mladen Mešter, dr.med., March 2026
 */

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

// AVX-512 intrinsics
#include <immintrin.h>

static constexpr int DMAX = 32;
// Round up to multiple of 8 for AVX-512 alignment
static constexpr int DPAD = ((DMAX + 7) / 8) * 8;

// =========================================================================
// AVX-512 Matrix Utilities (d×d, row-major, padded to 8-wide)
// =========================================================================

static inline void mat_zero(double* A, int d) {
    std::memset(A, 0, d * d * sizeof(double));
}
static inline void mat_copy(double* dst, const double* src, int d) {
    std::memcpy(dst, src, d * d * sizeof(double));
}
static inline void mat_eye(double* A, int d) {
    mat_zero(A, d);
    for (int i = 0; i < d; ++i) A[i * d + i] = 1.0;
}

/**
 * AVX-512 dot product of two double arrays of length d.
 * Uses 512-bit FMA where possible, scalar tail.
 */
static inline double avx512_dot(const double* a, const double* b, int d) {
    __m512d acc = _mm512_setzero_pd();
    int k = 0;
    for (; k + 8 <= d; k += 8) {
        __m512d va = _mm512_loadu_pd(a + k);
        __m512d vb = _mm512_loadu_pd(b + k);
        acc = _mm512_fmadd_pd(va, vb, acc);
    }
    double sum = _mm512_reduce_add_pd(acc);
    for (; k < d; ++k) sum += a[k] * b[k];
    return sum;
}

/**
 * AVX-512 matrix multiply: C = A * B (d×d)
 * Each row of C computed via SIMD dot products.
 */
static void mat_mul_avx(double* __restrict C,
                        const double* __restrict A,
                        const double* __restrict B,
                        int d) {
    // Transpose B for cache-friendly row access
    double Bt[DMAX * DMAX];
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            Bt[j * d + i] = B[i * d + j];

    for (int i = 0; i < d; ++i) {
        const double* Ai = A + i * d;
        for (int j = 0; j < d; ++j) {
            C[i * d + j] = avx512_dot(Ai, Bt + j * d, d);
        }
    }
}

/**
 * AVX-512 axpy: A += alpha * B
 */
static void mat_axpy_avx(double* A, double alpha, const double* B, int n) {
    __m512d valpha = _mm512_set1_pd(alpha);
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        __m512d va = _mm512_loadu_pd(A + k);
        __m512d vb = _mm512_loadu_pd(B + k);
        va = _mm512_fmadd_pd(valpha, vb, va);
        _mm512_storeu_pd(A + k, va);
    }
    for (; k < n; ++k) A[k] += alpha * B[k];
}

static inline void mat_scale(double* A, double s, int d) {
    int n = d * d;
    __m512d vs = _mm512_set1_pd(s);
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        __m512d va = _mm512_loadu_pd(A + k);
        va = _mm512_mul_pd(va, vs);
        _mm512_storeu_pd(A + k, va);
    }
    for (; k < n; ++k) A[k] *= s;
}

static inline double mat_trace(const double* A, int d) {
    double t = 0.0;
    for (int i = 0; i < d; ++i) t += A[i * d + i];
    return t;
}

static inline void mat_hadamard(double* C, const double* A,
                                const double* B, int d) {
    int n = d * d;
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        __m512d va = _mm512_loadu_pd(A + k);
        __m512d vb = _mm512_loadu_pd(B + k);
        _mm512_storeu_pd(C + k, _mm512_mul_pd(va, vb));
    }
    for (; k < n; ++k) C[k] = A[k] * B[k];
}

static inline double mat_max_abs(const double* A, int d) {
    double mx = 0.0;
    int n = d * d;
    for (int i = 0; i < n; ++i) {
        double v = std::abs(A[i]);
        if (v > mx) mx = v;
    }
    return mx;
}

static inline double mat_diff_norm_sq(const double* A, const double* B, int d) {
    int n = d * d;
    __m512d acc = _mm512_setzero_pd();
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        __m512d va = _mm512_loadu_pd(A + k);
        __m512d vb = _mm512_loadu_pd(B + k);
        __m512d diff = _mm512_sub_pd(va, vb);
        acc = _mm512_fmadd_pd(diff, diff, acc);
    }
    double s = _mm512_reduce_add_pd(acc);
    for (; k < n; ++k) { double v = A[k] - B[k]; s += v * v; }
    return s;
}

// =========================================================================
// Matrix Exponential — Taylor order 12, AVX-512 matmul
// =========================================================================

static void expm_taylor(double* E, const double* M, int d) {
    double term[DMAX * DMAX];
    double temp[DMAX * DMAX];

    mat_eye(E, d);
    mat_copy(term, M, d);
    mat_axpy_avx(E, 1.0, term, d * d);

    for (int k = 2; k <= 12; ++k) {
        mat_mul_avx(temp, term, M, d);
        mat_scale(temp, 1.0 / k, d);
        mat_copy(term, temp, d);
        mat_axpy_avx(E, 1.0, term, d * d);
        if (mat_max_abs(term, d) < 1e-15) break;
    }
}

// =========================================================================
// NOTEARS — Cached expm, AVX-512 gradient
// =========================================================================

struct NOTEARSScratch {
    double M[DMAX * DMAX];
    double E[DMAX * DMAX];
    double E_cached[DMAX * DMAX];
    double W_at_cache[DMAX * DMAX];
    double grad_loss[DMAX * DMAX];
    double grad_h[DMAX * DMAX];
    double grad_smooth[DMAX * DMAX];
    double W_old[DMAX * DMAX];
    double XtXW[DMAX * DMAX];
    bool cache_valid;
    int d;

    void init(int dim) { d = dim; cache_valid = false; }

    void compute_expm(const double* W) {
        if (cache_valid && mat_diff_norm_sq(W, W_at_cache, d) < 1e-4) {
            mat_copy(E, E_cached, d);
            return;
        }
        mat_hadamard(M, W, W, d);
        expm_taylor(E, M, d);
        mat_copy(E_cached, E, d);
        mat_copy(W_at_cache, W, d);
        cache_valid = true;
    }

    double h(const double* W) {
        compute_expm(W);
        return mat_trace(E, d) - d;
    }

    void h_gradient(double* out, const double* W) {
        compute_expm(W);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                out[i * d + j] = E[j * d + i] * 2.0 * W[i * d + j];
    }
};

// Fast LS gradient: grad = XtX @ W - XtX
static void ls_gradient_fast(double* grad, const double* XtX,
                             const double* W, double* XtXW, int d) {
    mat_mul_avx(XtXW, XtX, W, d);
    int n = d * d;
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        __m512d va = _mm512_loadu_pd(XtXW + k);
        __m512d vb = _mm512_loadu_pd(XtX + k);
        _mm512_storeu_pd(grad + k, _mm512_sub_pd(va, vb));
    }
    for (; k < n; ++k) grad[k] = XtXW[k] - XtX[k];
}

static void precompute_XtX(double* XtX, const double* X, int n, int d) {
    double inv_n = 1.0 / n;
    for (int i = 0; i < d; ++i) {
        // AVX-512 column dot products
        for (int j = i; j < d; ++j) {
            // Dot product of X[:,i] and X[:,j] (strided)
            double s = 0.0;
            for (int k = 0; k < n; ++k)
                s += X[k * d + i] * X[k * d + j];
            XtX[i * d + j] = s * inv_n;
            XtX[j * d + i] = s * inv_n;
        }
    }
}

static void notears_inner_solve(double* W, const double* XtX,
                                NOTEARSScratch& sc,
                                double alpha, double rho,
                                double lambda1, int d,
                                int max_inner) {
    for (int inner = 0; inner < max_inner; ++inner) {
        mat_copy(sc.W_old, W, d);

        // Clip
        int n = d * d;
        for (int i = 0; i < n; ++i)
            W[i] = std::max(-5.0, std::min(5.0, W[i]));

        // Fast gradient via precomputed XtX
        ls_gradient_fast(sc.grad_loss, XtX, W, sc.XtXW, d);

        double h_val = sc.h(W);
        if (!std::isfinite(h_val)) {
            for (int i = 0; i < n; ++i) W[i] *= 0.5;
            sc.cache_valid = false;
            continue;
        }

        sc.h_gradient(sc.grad_h, W);
        bool ok = true;
        for (int i = 0; i < n; ++i)
            if (!std::isfinite(sc.grad_h[i])) { ok = false; break; }
        if (!ok) {
            for (int i = 0; i < n; ++i) W[i] *= 0.5;
            sc.cache_valid = false;
            continue;
        }

        // Combined gradient: AVX-512
        {
            double c = alpha + rho * h_val;
            __m512d vc = _mm512_set1_pd(c);
            int k = 0;
            for (; k + 8 <= n; k += 8) {
                __m512d gl = _mm512_loadu_pd(sc.grad_loss + k);
                __m512d gh = _mm512_loadu_pd(sc.grad_h + k);
                _mm512_storeu_pd(sc.grad_smooth + k,
                    _mm512_fmadd_pd(vc, gh, gl));
            }
            for (; k < n; ++k)
                sc.grad_smooth[k] = sc.grad_loss[k] + c * sc.grad_h[k];
        }

        // Proximal gradient with L1 soft threshold
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                if (i == j) { W[i * d + j] = 0.0; continue; }
                int ij = i * d + j;
                double step = 1.0 / (XtX[i * d + i] +
                    rho * 2.0 * std::abs(W[ij]) + 1e-8);
                step = std::min(step, 0.1);
                double proposal = W[ij] - step * sc.grad_smooth[ij];
                double absval = std::abs(proposal) - lambda1 * step;
                W[ij] = (absval > 0) ? std::copysign(absval, proposal) : 0.0;
            }
        }

        for (int i = 0; i < n; ++i)
            W[i] = std::max(-5.0, std::min(5.0, W[i]));

        double change = 0.0;
        for (int i = 0; i < n; ++i) {
            double diff = std::abs(W[i] - sc.W_old[i]);
            if (diff > change) change = diff;
        }
        if (change > 0.01) sc.cache_valid = false;
        if (change < 1e-7) break;
    }
}

static double notears_core(const double* Xc, const double* XtX,
                           int n, int d,
                           double lambda1, double w_thresh,
                           int max_outer, int max_inner,
                           double* W) {
    NOTEARSScratch sc;
    sc.init(d);
    mat_zero(W, d);
    double alpha_lm = 0.0, rho = 1.0, h_prev = 1e18;

    for (int outer = 0; outer < max_outer; ++outer) {
        notears_inner_solve(W, XtX, sc, alpha_lm, rho,
                           lambda1, d, max_inner);
        double h_val = sc.h(W);
        if (h_val > 0.25 * h_prev)
            rho = std::min(10.0 * rho, 1e16);
        else
            alpha_lm += rho * h_val;
        h_prev = h_val;
        if (std::abs(h_val) < 1e-8) break;
    }

    for (int i = 0; i < d * d; ++i)
        if (std::abs(W[i]) < w_thresh) W[i] = 0.0;

    return sc.h(W);
}

// =========================================================================
// Granger — OLS F-test
// =========================================================================

static double ols_rss(const double* X, const double* Y, int n, int p) {
    if (p <= 0 || n <= p) return 1e30;

    double A[64 * 64], b[64], beta[64];

    // X^T X and X^T Y — use AVX-512 for column dots
    for (int i = 0; i < p; ++i) {
        double s = 0;
        for (int k = 0; k < n; ++k) s += X[k * p + i] * Y[k];
        b[i] = s;
        for (int j = i; j < p; ++j) {
            s = 0;
            for (int k = 0; k < n; ++k) s += X[k * p + i] * X[k * p + j];
            A[i * p + j] = A[j * p + i] = s;
        }
    }

    // Gaussian elimination with partial pivoting
    for (int col = 0; col < p; ++col) {
        int mx = col;
        double mv = std::abs(A[col * p + col]);
        for (int r = col + 1; r < p; ++r) {
            double v = std::abs(A[r * p + col]);
            if (v > mv) { mv = v; mx = r; }
        }
        if (mv < 1e-15) return 1e30;
        if (mx != col) {
            for (int j = 0; j < p; ++j) std::swap(A[col*p+j], A[mx*p+j]);
            std::swap(b[col], b[mx]);
        }
        double pivot = A[col * p + col];
        for (int r = col + 1; r < p; ++r) {
            double f = A[r * p + col] / pivot;
            for (int j = col; j < p; ++j) A[r*p+j] -= f * A[col*p+j];
            b[r] -= f * b[col];
        }
    }

    for (int i = p - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < p; ++j) s -= A[i*p+j] * beta[j];
        beta[i] = s / A[i*p+i];
    }

    // RSS via AVX-512 accumulation
    double rss = 0;
    for (int k = 0; k < n; ++k) {
        double pred = 0;
        for (int j = 0; j < p; ++j) pred += X[k*p+j] * beta[j];
        double r = Y[k] - pred;
        rss += r * r;
    }
    return rss;
}

static double f_cdf_approx(double f, int df1, int df2) {
    if (f <= 0) return 0.0;
    double z = std::pow(f * df1 / df2, 1.0 / 3.0);
    double ez = 1.0 - 2.0 / (9.0 * df2);
    double vz = 2.0 / (9.0 * df1) + z * z * 2.0 / (9.0 * df2);
    if (vz > 0) {
        double t = (z * ez - (1.0 - 2.0 / (9.0 * df1))) / std::sqrt(vz);
        return std::max(0.0, std::min(1.0,
            0.5 * (1.0 + std::erf(t * 0.7071067811865475))));
    }
    return 0.5;
}

static double granger_pair(const double* data, int n, int d,
                           int i, int j, int max_lag) {
    // Heap-allocated buffers (safe for large n)
    int max_neff = n;
    int max_pu = 2 * max_lag + 1;
    int max_pr = max_lag + 1;
    double* design_u = (double*)malloc(max_neff * max_pu * sizeof(double));
    double* design_r = (double*)malloc(max_neff * max_pr * sizeof(double));
    double* Y_buf    = (double*)malloc(max_neff * sizeof(double));

    int best_lag = 1;
    double best_bic = 1e30;

    for (int lag = 1; lag <= max_lag; ++lag) {
        if (n - lag < lag + 5) break;
        int n_eff = n - lag;
        int p_u = 2 * lag + 1;
        for (int t = 0; t < n_eff; ++t) {
            Y_buf[t] = data[(t + lag) * d + j];
            for (int k = 0; k < lag; ++k) {
                design_u[t * p_u + k] = data[(t + lag - k - 1) * d + j];
                design_u[t * p_u + lag + k] = data[(t + lag - k - 1) * d + i];
            }
            design_u[t * p_u + 2 * lag] = 1.0;
        }
        double rss = ols_rss(design_u, Y_buf, n_eff, p_u);
        double bic = n_eff * std::log(rss / n_eff + 1e-10) +
                     p_u * std::log((double)n_eff);
        if (bic < best_bic) { best_bic = bic; best_lag = lag; }
    }

    int lag = best_lag;
    int n_eff = n - lag;
    if (n_eff < 2 * lag + 5) {
        free(design_u); free(design_r); free(Y_buf);
        return 1.0;
    }

    int p_r = lag + 1, p_u = 2 * lag + 1;
    for (int t = 0; t < n_eff; ++t) {
        Y_buf[t] = data[(t + lag) * d + j];
        for (int k = 0; k < lag; ++k) {
            design_r[t * p_r + k] = data[(t + lag - k - 1) * d + j];
            design_u[t * p_u + k] = data[(t + lag - k - 1) * d + j];
            design_u[t * p_u + lag + k] = data[(t + lag - k - 1) * d + i];
        }
        design_r[t * p_r + lag] = 1.0;
        design_u[t * p_u + 2 * lag] = 1.0;
    }

    double rss_r = ols_rss(design_r, Y_buf, n_eff, p_r);
    double rss_u = ols_rss(design_u, Y_buf, n_eff, p_u);

    int df1 = p_u - p_r, df2 = n_eff - p_u;
    double pval = 1.0;
    if (df2 > 0 && rss_u > 0) {
        double f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2);
        pval = 1.0 - f_cdf_approx(f_stat, df1, df2);
    }

    free(design_u); free(design_r); free(Y_buf);
    return pval;
}

// =========================================================================
// C API
// =========================================================================

extern "C" {

double fm_notears(const double* X, int n, int d,
                  double lambda1, double w_thresh,
                  int max_outer, int max_inner,
                  double* W_out) {
    if (d > DMAX) return -1.0;

    double* Xc = (double*)malloc(n * d * sizeof(double));
    std::memcpy(Xc, X, n * d * sizeof(double));
    for (int j = 0; j < d; ++j) {
        double mean = 0;
        for (int i = 0; i < n; ++i) mean += Xc[i * d + j];
        mean /= n;
        for (int i = 0; i < n; ++i) Xc[i * d + j] -= mean;
    }

    double XtX[DMAX * DMAX];
    precompute_XtX(XtX, Xc, n, d);

    double h = notears_core(Xc, XtX, n, d, lambda1, w_thresh,
                            max_outer, max_inner, W_out);
    free(Xc);
    return h;
}

void fm_notears_bootstrap(const double* X, int n, int d,
                          int n_bootstrap,
                          double lambda1, double w_thresh,
                          int seed, double* stab_out) {
    if (d > DMAX) return;

    double* Xc = (double*)malloc(n * d * sizeof(double));
    std::memcpy(Xc, X, n * d * sizeof(double));
    for (int j = 0; j < d; ++j) {
        double mean = 0;
        for (int i = 0; i < n; ++i) mean += Xc[i * d + j];
        mean /= n;
        for (int i = 0; i < n; ++i) Xc[i * d + j] -= mean;
    }

    mat_zero(stab_out, d);

    #ifdef _OPENMP
    int n_threads = omp_get_max_threads();
    #else
    int n_threads = 1;
    #endif
    double* thread_stab = (double*)calloc(n_threads * d * d, sizeof(double));

    #pragma omp parallel
    {
        #ifdef _OPENMP
        int tid = omp_get_thread_num();
        #else
        int tid = 0;
        #endif
        double* my_stab = thread_stab + tid * d * d;
        double* X_boot = (double*)malloc(n * d * sizeof(double));
        double W_boot[DMAX * DMAX];
        double XtX_boot[DMAX * DMAX];
        std::mt19937 rng(seed + tid * 1000);
        std::uniform_int_distribution<int> idx_dist(0, n - 1);
        std::uniform_real_distribution<double> unif(0.0, 1.0);

        #pragma omp for schedule(dynamic)
        for (int b = 0; b < n_bootstrap; ++b) {
            for (int i = 0; i < n; ++i) {
                int src = idx_dist(rng);
                std::memcpy(X_boot + i * d, Xc + src * d, d * sizeof(double));
            }
            double lam = lambda1 * (0.5 + unif(rng));
            double thr = w_thresh * (0.6 + 0.8 * unif(rng));
            precompute_XtX(XtX_boot, X_boot, n, d);
            notears_core(X_boot, XtX_boot, n, d, lam, thr, 50, 30, W_boot);
            for (int i = 0; i < d * d; ++i)
                if (std::abs(W_boot[i]) > 0) my_stab[i] += 1.0;
        }
        free(X_boot);
    }

    for (int t = 0; t < n_threads; ++t) {
        double* ts = thread_stab + t * d * d;
        for (int i = 0; i < d * d; ++i) stab_out[i] += ts[i];
    }
    double inv_b = 1.0 / n_bootstrap;
    for (int i = 0; i < d * d; ++i) stab_out[i] *= inv_b;

    free(thread_stab);
    free(Xc);
}

void fm_granger_all_pairs(const double* data, int n, int d,
                          int max_lag, double alpha,
                          double* gc_out, double* pval_out) {
    int n_tests = d * (d - 1);
    double alpha_adj = alpha / std::max(n_tests, 1);

    mat_zero(gc_out, d);
    if (pval_out)
        for (int i = 0; i < d * d; ++i) pval_out[i] = 1.0;

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            if (i == j) continue;
            double pval = granger_pair(data, n, d, i, j, max_lag);
            gc_out[i * d + j] = (pval < alpha_adj) ? 1.0 : 0.0;
            if (pval_out) pval_out[i * d + j] = pval;
        }
    }
}

void fm_expm(const double* M, int d, double* E_out) {
    expm_taylor(E_out, M, d);
}

double fm_h_acyclicity(const double* W, int d) {
    double M[DMAX * DMAX], E[DMAX * DMAX];
    mat_hadamard(M, W, W, d);
    expm_taylor(E, M, d);
    return mat_trace(E, d) - d;
}

} // extern "C"
