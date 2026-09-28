// ============================================================
//  极致加速批量 SVD —— 自定义 Jacobi 核 vs cuSOLVER
//
//  目标场景: 大量 NxN 小矩阵 (N <= 32) 同时 SVD 分解
//           典型用例: ICP / 本质矩阵 / 协方差对齐 / PCA 等
//
//  自定义核优势:
//    1) 一个 block 处理一个矩阵 —— 零跨矩阵同步
//    2) 全程驻留 shared memory —— 零全局内存往返
//    3) 单 warp + warp-shuffle 归约 —— 无 __syncthreads()
//    4) 编译期 N + 全展开 —— 零分支调度开销
// ============================================================

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t e = (call);                                                   \
        if (e != cudaSuccess) {                                                   \
            std::fprintf(stderr, "[CUDA] %s:%d %s\n",                             \
                         __FILE__, __LINE__, cudaGetErrorString(e));              \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

#define CUSOLVER_CHECK(call)                                                      \
    do {                                                                          \
        cusolverStatus_t s = (call);                                              \
        if (s != CUSOLVER_STATUS_SUCCESS) {                                       \
            std::fprintf(stderr, "[cuSOLVER] %s:%d status=%d\n",                  \
                         __FILE__, __LINE__, (int)s);                             \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)


// ============================================================
//  Warp-级 32 lane 求和（输入: 每个 lane 一个值；输出: 所有 lane 都拿到总和）
// ============================================================
__device__ __forceinline__ float warpSum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}


// ============================================================
//  核心 kernel: 批量 NxN one-sided Jacobi SVD
//  约束: 2 <= N <= 32
//  输入:  A  (batch, N, N)  列主序, device
//  输出:  U  (batch, N, N)  列主序, 左奇异向量
//         S  (batch, N)      奇异值, 已降序排序
//         V  (batch, N, N)  列主序, 右奇异向量 (注意: 不是 V^T)
//
//  算法:
//    用 cyclic-by-rows Jacobi 对 A 的列做正交化:
//    每次取列对 (p, q), 构造 Givens 旋转 R, 使得
//      [A_p A_q] R 的两列正交
//    迭代到收敛后:
//      列 i 的范数即为奇异值 S[i]
//      列 i 归一化即得 U 的列 i
//      累乘 R 即得 V
// ============================================================
template <int N>
__launch_bounds__(32)
__global__ void svdJacobiSmallKernel(
    const float* __restrict__ A,
    float* __restrict__ U,
    float* __restrict__ S,
    float* __restrict__ V,
    int batch)
{   
    //断言,判断 N 是否在 [2, 32] 范围内
    static_assert(N >= 2 && N <= 32, "N must be in [2, 32]");
    //防止线程块索引超出批量范围
    const int bid = blockIdx.x;
    if (bid >= batch) return;
    const int tid = threadIdx.x;
    //共享内存分配
    __shared__ float sA[N * N];
    __shared__ float sV[N * N];
    __shared__ float sS[N];
    __shared__ int   perm[N];

    // ---- 1. 载入 A (列主序), 初始化 V = I ----
    #pragma unroll
    for (int i = tid; i < N * N; i += 32) {
        //不是矩阵分块,而是把所有数据搬运到共享内存中,以便后续计算
        sA[i] = A[bid * N * N + i];
        //初始化 V 为单位矩阵
        sV[i] = ((i % N) == (i / N)) ? 1.f : 0.f;
    }
    __syncwarp();

    // ---- 2. cyclic-by-rows Jacobi sweeps ----
    constexpr int MAX_SWEEPS = 12;
    constexpr float TOL_SQ   = 1e-12f;

    for (int sw = 0; sw < MAX_SWEEPS; ++sw) {
        float off_sq = 0.f;

        #pragma unroll
        for (int p = 0; p < N - 1; ++p) {
            #pragma unroll
            for (int q = p + 1; q < N; ++q) {
                // 取列 p, q 的第 tid 行 (tid >= N 时填 0, 不影响归约)
                float ap = (tid < N) ? sA[p * N + tid] : 0.f;
                float aq = (tid < N) ? sA[q * N + tid] : 0.f;
                //单warp 归约求和, 得到列 p, q 的范数平方和以及内积
                float alpha = warpSum(ap * ap);
                float beta  = warpSum(aq * aq);
                float gamma = warpSum(ap * aq);

                // 计算 Givens 旋转参数 (c, s)
                float c = 1.f, s = 0.f;
                if (fabsf(gamma) > 1e-30f * fmaxf(fabsf(alpha), fabsf(beta))) {
                    float zeta = (beta - alpha) / (2.f * gamma);
                    float t = (zeta >= 0.f)
                        ? 1.f / (zeta + sqrtf(zeta * zeta + 1.f))
                        : 1.f / (zeta - sqrtf(zeta * zeta + 1.f));
                    c = rsqrtf(1.f + t * t);
                    s = c * t;
                    off_sq += gamma * gamma;   // 所有 lane 拿到的 gamma 相同, 故 off_sq 一致
                }

                // 应用旋转: 同时更新 A 和 V 的第 p, q 列
                if (tid < N) {
                    float vp = sV[p * N + tid];
                    float vq = sV[q * N + tid];
                    sA[p * N + tid] = c * ap - s * aq;
                    sA[q * N + tid] = s * ap + c * aq;
                    sV[p * N + tid] = c * vp - s * vq;
                    sV[q * N + tid] = s * vp + c * vq;
                }
                __syncwarp();
            }
        }

        if (off_sq < TOL_SQ) break;
    }

    __syncwarp();

    // ---- 3. 计算 S[i] = ||A[:, i]|| ----
    if (tid < N) {
        float ns = 0.f;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            float v = sA[tid * N + i];
            ns += v * v;
        }
        sS[tid] = sqrtf(ns);
    }
    __syncwarp();

    // ---- 4. 单线程做选择排序, 得到降序排列 ----
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < N; ++i) perm[i] = i;
        for (int i = 0; i < N - 1; ++i) {
            int idx = i;
            for (int j = i + 1; j < N; ++j)
                if (sS[perm[j]] > sS[perm[idx]]) idx = j;
            if (idx != i) {
                int tmp = perm[i];
                perm[i] = perm[idx];
                perm[idx] = tmp;
            }
        }
    }
    __syncwarp();

    // ---- 5. 写出排序后的 U / S / V ----
    if (tid < N) {
        const int src = perm[tid];
        const float sval = sS[src];
        const float inv  = (sval > 1e-30f) ? 1.f / sval : 0.f;
        S[bid * N + tid] = sval;
        #pragma unroll
        for (int r = 0; r < N; ++r) {
            U[bid * N * N + tid * N + r] = sA[r + src * N] * inv;
            V[bid * N * N + tid * N + r] = sV[r + src * N];
        }
    }
}


// ============================================================
//  基准: 自定义核 vs cuSOLVER gesvdjBatched
//  时间均不含 H2D/D2H, 仅 GPU 计算时间
//  重构误差: ||A - U * diag(S) * V^T||_F / ||A||_F  (前 32 个样本)
// ============================================================
template <int N>
void benchVsCusolver(int batch, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> hA(static_cast<size_t>(N) * N * batch);
    for (auto& v : hA) v = dist(rng);

    float *dA, *dA_cp, *dU, *dS, *dV;
    CUDA_CHECK(cudaMalloc(&dA,    sizeof(float) * N * N * batch));
    CUDA_CHECK(cudaMalloc(&dA_cp, sizeof(float) * N * N * batch));
    CUDA_CHECK(cudaMalloc(&dU,    sizeof(float) * N * N * batch));
    CUDA_CHECK(cudaMalloc(&dS,    sizeof(float) * N * batch));
    CUDA_CHECK(cudaMalloc(&dV,    sizeof(float) * N * N * batch));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(),
                          sizeof(float) * N * N * batch, cudaMemcpyHostToDevice));

    constexpr int ITERS = 10;

    // ---------- 自定义核 ----------
    // 预热
    svdJacobiSmallKernel<N><<<batch, 32>>>(dA, dU, dS, dV, batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < ITERS; ++i)
        svdJacobiSmallKernel<N><<<batch, 32>>>(dA, dU, dS, dV, batch);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float t_custom; cudaEventElapsedTime(&t_custom, a, b);
    t_custom /= ITERS;
    cudaEventDestroy(a); cudaEventDestroy(b);

    std::vector<float> hU1(static_cast<size_t>(N)*N*batch);
    std::vector<float> hS1(static_cast<size_t>(N)*batch);
    std::vector<float> hV1(static_cast<size_t>(N)*N*batch);
    cudaMemcpy(hU1.data(), dU, sizeof(float)*N*N*batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(hS1.data(), dS, sizeof(float)*N*batch,   cudaMemcpyDeviceToHost);
    cudaMemcpy(hV1.data(), dV, sizeof(float)*N*N*batch, cudaMemcpyDeviceToHost);

    // ---------- cuSOLVER gesvdjBatched ----------
    cusolverDnHandle_t h;
    CUSOLVER_CHECK(cusolverDnCreate(&h));
    gesvdjInfo_t pp;
    CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&pp));
    CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(pp, 1e-7));
    CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(pp, 15));

    int* dInfo; CUDA_CHECK(cudaMalloc(&dInfo, sizeof(int) * batch));
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgesvdjBatched_bufferSize(
        h, CUSOLVER_EIG_MODE_VECTOR, N, N,
        dA_cp, N, dS, dU, N, dV, N, &lwork, pp, batch));
    float* dWork; CUDA_CHECK(cudaMalloc(&dWork, sizeof(float) * lwork));

    // 预热
    CUDA_CHECK(cudaMemcpy(dA_cp, dA, sizeof(float)*N*N*batch, cudaMemcpyDeviceToDevice));
    CUSOLVER_CHECK(cusolverDnSgesvdjBatched(
        h, CUSOLVER_EIG_MODE_VECTOR, N, N,
        dA_cp, N, dS, dU, N, dV, N, dWork, lwork, dInfo, pp, batch));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时: 每次迭代独立计时, 不含 memcpy
    float t_cusolver = 0.f;
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaMemcpy(dA_cp, dA,
                              sizeof(float)*N*N*batch, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t aa, bb;
        cudaEventCreate(&aa); cudaEventCreate(&bb);
        cudaEventRecord(aa);
        CUSOLVER_CHECK(cusolverDnSgesvdjBatched(
            h, CUSOLVER_EIG_MODE_VECTOR, N, N,
            dA_cp, N, dS, dU, N, dV, N, dWork, lwork, dInfo, pp, batch));
        cudaEventRecord(bb);
        cudaEventSynchronize(bb);
        float dt; cudaEventElapsedTime(&dt, aa, bb);
        t_cusolver += dt;
        cudaEventDestroy(aa); cudaEventDestroy(bb);
    }
    t_cusolver /= ITERS;

    std::vector<float> hU2(static_cast<size_t>(N)*N*batch);
    std::vector<float> hS2(static_cast<size_t>(N)*batch);
    std::vector<float> hV2(static_cast<size_t>(N)*N*batch);
    cudaMemcpy(hU2.data(), dU, sizeof(float)*N*N*batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(hS2.data(), dS, sizeof(float)*N*batch,   cudaMemcpyDeviceToHost);
    cudaMemcpy(hV2.data(), dV, sizeof(float)*N*N*batch, cudaMemcpyDeviceToHost);

    // ---------- 重构误差 ----------
    auto recErr = [&](const std::vector<float>& U,
                      const std::vector<float>& S,
                      const std::vector<float>& V) -> double {
        double num = 0, den = 0;
        const int K = std::min(batch, 32);
        for (int bb = 0; bb < K; ++bb) {
            for (int r = 0; r < N; ++r) {
                for (int c = 0; c < N; ++c) {
                    double acc = 0;
                    for (int k = 0; k < N; ++k) {
                        acc += (double)U[bb*N*N + r + k*N]
                             * (double)S[bb*N + k]
                             * (double)V[bb*N*N + c + k*N];
                    }
                    double diff = (double)hA[bb*N*N + r + c*N] - acc;
                    num += diff * diff;
                    den += (double)hA[bb*N*N + r + c*N]
                         * (double)hA[bb*N*N + r + c*N];
                }
            }
        }
        return std::sqrt(num / (den + 1e-30));
    };

    double e_custom   = recErr(hU1, hS1, hV1);
    double e_cusolver = recErr(hU2, hS2, hV2);

    std::printf("N=%2d  batch=%7d | custom=%8.3f ms  cuSOLVER=%8.3f ms  speedup=%6.2fx | err: custom=%.2e  cuSOLVER=%.2e\n",
                N, batch, t_custom, t_cusolver, t_cusolver / t_custom,
                e_custom, e_cusolver);

    cudaFree(dA); cudaFree(dA_cp); cudaFree(dU); cudaFree(dS); cudaFree(dV);
    cudaFree(dInfo); cudaFree(dWork);
    cusolverDnDestroyGesvdjInfo(pp);
    cusolverDnDestroy(h);
}


int main()
{
    std::printf("========== 极致加速批量 SVD: 自定义核 vs cuSOLVER ==========\n");

    std::printf("\n[中等 batch]\n");
    benchVsCusolver<3>(10000, 42);
    benchVsCusolver<4>(10000, 42);
    benchVsCusolver<8>(10000, 42);
    benchVsCusolver<16>(10000, 42);
    benchVsCusolver<32>(10000, 42);

    std::printf("\n[大 batch]\n");
    benchVsCusolver<3>(100000, 42);
    benchVsCusolver<4>(100000, 42);
    benchVsCusolver<8>(100000, 42);
    benchVsCusolver<16>(100000, 42);

    std::printf("\n[超大 batch]\n");
    benchVsCusolver<3>(1000000, 42);
    benchVsCusolver<4>(1000000, 42);

    std::printf("\n==========\n");
    return 0;
}