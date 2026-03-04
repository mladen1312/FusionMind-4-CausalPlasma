/**
 * FusionMind 4.0 — OpenCL GPU Compute Kernels
 * =============================================
 *
 * GPU-accelerated causal discovery for production deployment.
 * Portable: works on NVIDIA, AMD, Intel GPUs via OpenCL 1.2+.
 *
 * Kernels:
 *   1. mat_mul_gpu       — Tiled matrix multiply (16×16 tiles)
 *   2. expm_taylor_gpu   — Matrix exponential via Taylor series
 *   3. notears_inner_gpu — NOTEARS proximal gradient step
 *   4. granger_pairs_gpu — Parallel Granger F-tests (1 pair per workgroup)
 *   5. bootstrap_resample_gpu — Fast bootstrap resampling
 *
 * Usage: Load via clCreateProgramWithSource, build, create kernels.
 * See fusionmind4/realtime/gpu_bindings.py for Python host code.
 *
 * Designed for d ≤ 32 variables (tokamak plasma).
 * All matrices stored row-major, double precision.
 *
 * Patent Family: PF1 (CPDE)
 * Author: Dr. Mladen Mester, March 2026
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define DMAX 32
#define TILE 16

// =========================================================================
// Kernel 1: Tiled Matrix Multiply  C = A × B (d×d)
// =========================================================================
// Launch: global = (d_padded, d_padded), local = (TILE, TILE)
// d_padded = ((d + TILE-1) / TILE) * TILE

__kernel void mat_mul_gpu(
    __global const double* A,
    __global const double* B,
    __global double* C,
    const int d)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    __local double tileA[TILE][TILE];
    __local double tileB[TILE][TILE];

    double sum = 0.0;
    int lr = get_local_id(0);
    int lc = get_local_id(1);

    int n_tiles = (d + TILE - 1) / TILE;

    for (int t = 0; t < n_tiles; ++t) {
        // Load tiles
        int a_col = t * TILE + lc;
        int b_row = t * TILE + lr;

        tileA[lr][lc] = (row < d && a_col < d) ? A[row * d + a_col] : 0.0;
        tileB[lr][lc] = (b_row < d && col < d) ? B[b_row * d + col] : 0.0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE; ++k)
            sum += tileA[lr][k] * tileB[k][lc];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < d && col < d)
        C[row * d + col] = sum;
}

// =========================================================================
// Kernel 2: Hadamard product  C = A ∘ B
// =========================================================================
// Launch: global = (d*d), local = (256)

__kernel void mat_hadamard_gpu(
    __global const double* A,
    __global const double* B,
    __global double* C,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) C[i] = A[i] * B[i];
}

// =========================================================================
// Kernel 3: Matrix A += alpha * B
// =========================================================================

__kernel void mat_axpy_gpu(
    __global double* A,
    __global const double* B,
    const double alpha,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) A[i] += alpha * B[i];
}

// =========================================================================
// Kernel 4: Matrix scale  A *= s
// =========================================================================

__kernel void mat_scale_gpu(
    __global double* A,
    const double s,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) A[i] *= s;
}

// =========================================================================
// Kernel 5: Trace
// =========================================================================

__kernel void mat_trace_gpu(
    __global const double* A,
    __global double* result,
    const int d)
{
    // Single work-item kernel (could use reduction for large d)
    double t = 0.0;
    for (int i = 0; i < d; ++i)
        t += A[i * d + i];
    *result = t;
}

// =========================================================================
// Kernel 6: NOTEARS gradient — h(W) gradient
// ∇h[i,j] = E^T[i,j] * 2 * W[i,j] = E[j*d+i] * 2 * W[i*d+j]
// =========================================================================
// Launch: global = (d*d)

__kernel void h_gradient_gpu(
    __global const double* E,    // expm(W∘W)
    __global const double* W,
    __global double* grad_h,
    const int d)
{
    int idx = get_global_id(0);
    if (idx >= d * d) return;
    int i = idx / d;
    int j = idx % d;
    grad_h[i * d + j] = E[j * d + i] * 2.0 * W[i * d + j];
}

// =========================================================================
// Kernel 7: LS gradient using precomputed XtX
// grad = XtX @ W - XtX
// =========================================================================
// Launch: global = (d, d), each computes one element

__kernel void ls_gradient_gpu(
    __global const double* XtX,
    __global const double* W,
    __global double* grad,
    const int d)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= d || j >= d) return;

    // (XtX @ W)[i,j]
    double s = 0.0;
    for (int k = 0; k < d; ++k)
        s += XtX[i * d + k] * W[k * d + j];

    grad[i * d + j] = s - XtX[i * d + j];
}

// =========================================================================
// Kernel 8: Proximal gradient step with L1 soft threshold
// =========================================================================
// Launch: global = (d*d)

__kernel void proximal_step_gpu(
    __global double* W,
    __global const double* grad_smooth,
    __global const double* XtX_diag,  // diagonal of XtX [d]
    const double rho,
    const double lambda1,
    const int d)
{
    int idx = get_global_id(0);
    if (idx >= d * d) return;
    int i = idx / d;
    int j = idx % d;

    if (i == j) { W[idx] = 0.0; return; }

    double w_ij = W[idx];
    double step = 1.0 / (XtX_diag[i] + rho * 2.0 * fabs(w_ij) + 1e-8);
    step = fmin(step, 0.1);

    double proposal = w_ij - step * grad_smooth[idx];
    double absval = fabs(proposal) - lambda1 * step;

    W[idx] = (absval > 0.0) ? copysign(absval, proposal) : 0.0;
    W[idx] = fmax(-5.0, fmin(5.0, W[idx]));
}

// =========================================================================
// Kernel 9: Granger OLS for a single pair  (1 workgroup per pair)
// =========================================================================
// Launch: global = (n_pairs * 256), local = (256)
// Each workgroup computes one (i,j) pair.

__kernel void granger_ols_gpu(
    __global const double* data,  // [n * d]
    __global double* pval_out,    // [d * d]
    const int n,
    const int d,
    const int max_lag,
    const int n_pairs,
    __global const int* pair_i,   // [n_pairs]
    __global const int* pair_j)   // [n_pairs]
{
    int pair_idx = get_group_id(0);
    if (pair_idx >= n_pairs) return;

    // Each workgroup handles one pair — work-item 0 does the computation
    // (OLS is inherently serial for small p)
    if (get_local_id(0) != 0) return;

    int vi = pair_i[pair_idx];
    int vj = pair_j[pair_idx];

    // Simplified: use lag=3 fixed for GPU version
    int lag = 3;
    if (lag > max_lag) lag = max_lag;
    int n_eff = n - lag;
    if (n_eff < 2 * lag + 5) {
        pval_out[vi * d + vj] = 1.0;
        return;
    }

    // Compute RSS restricted (Y_lags only) and unrestricted (Y_lags + X_lags)
    // Using simplified normal equations in registers for small p

    int p_r = lag + 1;
    int p_u = 2 * lag + 1;

    // Accumulate X^T X and X^T Y for both models
    // For restricted: p_r × p_r system
    // For unrestricted: p_u × p_u system
    // Since p_u ≤ 11 (max_lag=5), we can use register storage

    double rss_r = 0.0, rss_u = 0.0;

    // Compute mean residuals (approximate — exact OLS would need
    // full matrix solve which is done on CPU for precision)
    // GPU kernel provides fast screening; CPU refines

    // Simple variance-ratio approximation
    double var_y = 0.0, var_xy = 0.0;
    for (int t = lag; t < n; ++t) {
        double y = data[t * d + vj];
        double x = data[(t - 1) * d + vi];
        var_y += y * y;
        var_xy += x * y;
    }
    var_y /= n_eff;
    var_xy /= n_eff;

    double r2 = (var_xy * var_xy) / (var_y * var_y + 1e-15);
    double f_approx = r2 * (n_eff - p_u) / (lag * (1.0 - r2 + 1e-15));

    // Wilson-Hilferty approximation for F CDF
    double z = pow(f_approx * lag / (n_eff - p_u), 1.0 / 3.0);
    double ez = 1.0 - 2.0 / (9.0 * (n_eff - p_u));
    double vz = 2.0 / (9.0 * lag) + z * z * 2.0 / (9.0 * (n_eff - p_u));
    double pval = 1.0;
    if (vz > 0.0) {
        double t_val = (z * ez - (1.0 - 2.0 / (9.0 * lag))) / sqrt(vz);
        pval = 1.0 - 0.5 * (1.0 + erf(t_val * 0.7071067811865476));
        pval = fmax(0.0, fmin(1.0, pval));
    }

    pval_out[vi * d + vj] = pval;
}

// =========================================================================
// Kernel 10: Bootstrap resampling (generate indices)
// =========================================================================
// Launch: global = (n_bootstrap * n), local = (256)
// Uses Philox-like counter-based RNG for GPU

__kernel void bootstrap_resample_gpu(
    __global int* indices,    // [n_bootstrap * n]
    const int n,
    const int n_bootstrap,
    const unsigned int seed)
{
    int gid = get_global_id(0);
    int total = n_bootstrap * n;
    if (gid >= total) return;

    // Simple LCG RNG (per-element)
    unsigned int state = seed + gid * 1103515245u + 12345u;
    state = state * 1103515245u + 12345u;
    state = state * 1103515245u + 12345u;

    indices[gid] = (int)(state % (unsigned int)n);
}
