# SVD.cu — 批量小矩阵 SVD 加速库接口文档

> 文件: `SVD.cu`
> 依赖: CUDA Runtime, cuSOLVER (`cusolverDn.h`), cuBLAS (链接即可)
> 编译期最低要求: CUDA 11.0+, C++17, 计算能力 `sm_50` 及以上

---

## 目录

- [1. 概述](#1-概述)
- [2. 数学约定与内存布局](#2-数学约定与内存布局)
- [3. 编译](#3-编译)
- [4. 错误检查宏](#4-错误检查宏)
- [5. 设备端工具函数](#5-设备端工具函数)
- [6. 核心内核 `svdJacobiSmallKernel`](#6-核心内核-svdjacobismallkernel)
- [7. 基准接口 `benchVsCusolver`](#7-基准接口-benchvscusolver)
- [8. 算法原理](#8-算法原理)
- [9. 性能数据](#9-性能数据)
- [10. 使用范例](#10-使用范例)
- [11. 适用边界与已知限制](#11-适用边界与已知限制)
- [12. FAQ](#12-faq)

---

## 1. 概述

本库提供一个 **极致加速的批量小矩阵 SVD 内核**，针对 N×N 方阵 (2 ≤ N ≤ 32) 的大批量分解场景做了深度优化。

**核心场景**：
- 计算机视觉：批量 3×3 SVD（ICP 配准、本质矩阵分解、Procrustes 对齐）
- 机器人 / SLAM：协方差矩阵主轴分解
- 信号处理：小窗口 PCA
- 任意需要数千~百万个相同尺寸小矩阵同时 SVD 的场景

**与官方库的对比**（实测，sm_75 GPU）：

| N | batch | 本库 | cuSOLVER `gesvdjBatched` | 加速比 |
|---|---|---|---|---|
| 3 | 1,000,000 | 4.83 ms | 556 ms | **115×** |
| 4 | 1,000,000 | 11.2 ms | 617 ms | **55×** |
| 8 | 100,000 | 6.51 ms | 95.9 ms | **14.7×** |
| 16 | 100,000 | 57.3 ms | 137 ms | **2.4×** |
| 32 | 10,000 | 29.0 ms | 27.5 ms | 0.95× (持平) |

> N=32 是当前 single-warp 实现的天花板，再上去需要切换到 multi-warp 架构。

---

## 2. 数学约定与内存布局

### 2.1 SVD 定义

对每个矩阵 `A ∈ ℝ^(N×N)`，求分解：

```
A = U · diag(S) · V^T
```

其中：
- `U ∈ ℝ^(N×N)`：左奇异向量，列正交
- `S ∈ ℝ^N`：奇异值，**严格降序排列**
- `V ∈ ℝ^(N×N)`：右奇异向量，列正交（**注意输出的是 V，不是 V^T**）

### 2.2 内存布局

所有矩阵均使用 **列主序 (column-major)** 存储，与 cuSOLVER / LAPACK / cuBLAS 约定一致：

```
A[r, c]  →  A_flat[r + c * N]
```

### 2.3 批量布局

批量数据按 **「矩阵在前，元素在后」** 紧密排列：

```
A_batch[b * N * N + (r + c * N)]   // 第 b 个矩阵的 (r, c) 元素
S_batch[b * N + i]                 // 第 b 个矩阵的第 i 个奇异值
```

无 padding，无 stride。

---

## 3. 编译

### 3.1 直接编译（Windows + MSVC）

```bash
nvcc -O3 -std=c++17 -arch=sm_75 \
     -allow-unsupported-compiler \
     -Xcompiler /utf-8 \
     --use_fast_math \
     SVD.cu -lcusolver -lcublas -lcudart -o svd_test.exe
```

> Git Bash 用户：前置 `MSYS_NO_PATHCONV=1` 避免 `/utf-8` 被错误路径转换。

### 3.2 CMake 集成

`CMakeLists.txt` 中需链接 cuSOLVER（本仓库已配好）：

```cmake
target_link_libraries(${PROJECT_NAME} PRIVATE
        cudart
        cublas
        cusolver
)
```

### 3.3 编译选项推荐

| 选项 | 作用 |
|---|---|
| `-O3` | 启用全部优化 |
| `--use_fast_math` | 启用 `rsqrtf`、`sqrtf` 等的快速实现 |
| `-arch=sm_XX` | 指定计算能力（务必与运行 GPU 匹配） |
| `-Xcompiler /utf-8` | MSVC 下避免中文注释被 GBK 编码解析出错 |

---

## 4. 错误检查宏

### `CUDA_CHECK(call)`

封装 CUDA Runtime 调用，失败时打印文件、行号、错误信息后 `exit(1)`。

```cpp
CUDA_CHECK(cudaMalloc(&dA, sizeof(float) * 100));
```

### `CUSOLVER_CHECK(call)`

同上，封装 cuSOLVER 调用。

```cpp
CUSOLVER_CHECK(cusolverDnCreate(&handle));
```

> 生产环境建议替换为更友好的错误传播机制（异常 / 返回码），不要直接 `exit`。

---

## 5. 设备端工具函数

### `warpSum`

```cpp
__device__ __forceinline__ float warpSum(float v);
```

**功能**：对一个 warp（32 lanes）内每个 lane 持有的 `float` 做求和，归约结果广播到所有 lane。

**参数**：
- `v` — 当前 lane 持有的标量值

**返回**：所有 32 lanes 的 `v` 之和（每个 lane 拿到的值相同）。

**实现**：5 次 `__shfl_xor_sync`，零 shared memory 流量。

**约束**：调用时必须 **全部 32 lanes 同时进入**（不能在 divergent 分支里调用，否则未定义行为）。

---

## 6. 核心内核 `svdJacobiSmallKernel`

### 6.1 签名

```cpp
template <int N>
__launch_bounds__(32)
__global__ void svdJacobiSmallKernel(
    const float* __restrict__ A,
    float* __restrict__ U,
    float* __restrict__ S,
    float* __restrict__ V,
    int batch);
```

### 6.2 模板参数

| 参数 | 类型 | 范围 | 说明 |
|---|---|---|---|
| `N` | `int` | `[2, 32]` | 矩阵维度。**编译期常量**，因此每个 N 都会实例化一个独立的 kernel |

> 编译期错误：`N < 2` 或 `N > 32` 触发 `static_assert`。

### 6.3 运行时参数

| 参数 | 方向 | 形状（列主序） | 元素数 |
|---|---|---|---|
| `A` | 输入 | `batch × N × N` | `batch * N * N` |
| `U` | 输出 | `batch × N × N` | `batch * N * N` |
| `S` | 输出 | `batch × N` | `batch * N` |
| `V` | 输出 | `batch × N × N` | `batch * N * N` |
| `batch` | 输入 | 标量 | — |

所有指针 **必须指向 device memory**。

### 6.4 启动配置

**必须** 使用如下配置：

```cpp
dim3 grid(batch);
dim3 block(32);              // 必须是 32（单 warp）
svdJacobiSmallKernel<N><<<grid, block>>>(dA, dU, dS, dV, batch);
```

- `gridDim.x` = 矩阵数量；每个 block 处理一个矩阵
- `blockDim.x` 固定为 32（即一个 warp），**不要修改**

### 6.5 输出语义

- `S[b * N + i]` ≥ 0，且 `S[b*N+0] ≥ S[b*N+1] ≥ ... ≥ S[b*N+N-1]`
- `U` 的列是左奇异向量，对应 `S` 中的奇异值
- `V` 的列是右奇异向量，**注意是 V 而不是 V^T**
- 当某个奇异值小于 `1e-30` 时，对应 `U` 的列被置零（数值退化处理）

### 6.6 数值参数（kernel 内常量）

| 名称 | 值 | 含义 |
|---|---|---|
| `MAX_SWEEPS` | `12` | Jacobi 最大扫描轮数 |
| `TOL_SQ` | `1e-12f` | 提前退出的 off-diagonal 平方和阈值 |

> 修改这两个值会直接影响精度 / 速度权衡。

### 6.7 资源使用

| 资源 | 用量 |
|---|---|
| Shared memory / block | `2 * N * N * 4` + `N * 4` + `N * 4` 字节 ≈ `8 N² + 8 N` 字节 |
| Registers / thread | 取决于 N，N=32 约 32-40 个 |
| 单 SM 并发 block 数 | 取决于 GPU，sm_75 上 N=8 约 32 个/SM |

例：N=8 时每 block 占用 `8*64+8*8 = 576` 字节 shared，约 96 KB / SM 可装 ~160 块（远高于硬件每 SM 32 块上限）。

---

## 7. 基准接口 `benchVsCusolver`

### 7.1 签名

```cpp
template <int N>
void benchVsCusolver(int batch, uint32_t seed);
```

### 7.2 功能

1. 用 `seed` 生成 `batch` 个 N×N 随机矩阵（元素 ~U[-1, 1]）
2. 上传到 GPU，分别用 **自定义核** 与 **cuSOLVER `gesvdjBatched`** 各跑 10 次
3. 每次独立计时（不含 H2D/D2H、不含 device-to-device 拷贝），取平均
4. 对前 32 个样本计算重构相对 Frobenius 误差 `||A - U·diag(S)·V^T||_F / ||A||_F`
5. 打印一行结果：

```
N= 8  batch= 100000 | custom=   6.509 ms  cuSOLVER=  95.898 ms  speedup= 14.73x | err: custom=1.15e-06  cuSOLVER=1.77e-06
```

### 7.3 参数

| 参数 | 含义 |
|---|---|
| `batch` | 矩阵数量。建议 ≥ 1000 才能跑满 GPU |
| `seed` | 随机种子。相同 seed 保证两次运行得到相同输入，便于对比 |

### 7.4 注意

- cuSOLVER 的计时显式跳过了 `cudaMemcpy(dA_cp, dA, ...)` 的 device-to-device 拷贝（因为 `gesvdj` 会破坏输入），保证对比公平
- 内部使用 `cudaEvent` 计时，精度 ~0.5 μs

---

## 8. 算法原理

### 8.1 One-sided Jacobi SVD

设 `A ∈ ℝ^(N×N)`，目标是找到正交矩阵 `V` 使得 `A·V` 的列两两正交。一旦正交：

```
A · V = U · diag(S)       (列归一化即得 U, S)
A = U · diag(S) · V^T
```

### 8.2 cyclic-by-rows scheme

每个 sweep 遍历所有列对 `(p, q), p < q`，对每对计算 Givens 旋转：

```
α = ⟨A_p, A_p⟩,  β = ⟨A_q, A_q⟩,  γ = ⟨A_p, A_q⟩
ζ = (β - α) / (2γ)
t = sign(ζ) / (|ζ| + √(ζ²+1))
c = 1 / √(1 + t²),  s = c · t
[A_p, A_q] ← [c·A_p - s·A_q, s·A_p + c·A_q]
[V_p, V_q] ← 同样旋转
```

收敛后 `off_sq = ∑ γ²` 接近 0。

### 8.3 优化点

| 优化 | 实现位置 | 收益 |
|---|---|---|
| 一 block 一矩阵 | `<<<batch, 32>>>` | 零跨矩阵同步 |
| Shared memory 全程驻留 | `sA`, `sV`, `sS`, `perm` | 0 次全局内存往返 |
| Warp shuffle 归约 | `warpSum` | 比 shared memory 归约快 ~3 倍 |
| 编译期 `N` + `#pragma unroll` | 模板化 | 循环开销归零 |
| `__syncwarp` 而非 `__syncthreads` | 整个 kernel | 同步开销减少 ~5 倍 |
| 单线程做尾部排序 | `tid == 0` 块 | N² 比较，远低于 SVD 主体计算 |

---

## 9. 性能数据

### 9.1 测试环境

- GPU: sm_75 (RTX 20xx 级)
- 驱动 / CUDA: 12.8
- 编译: `-O3 --use_fast_math`
- 测量: 10 次平均，剔除首次预热

### 9.2 完整数据

```
[中等 batch]
N= 3  batch=  10000 | custom=  0.058 ms  cuSOLVER=   5.657 ms  speedup= 97.24x
N= 4  batch=  10000 | custom=  0.110 ms  cuSOLVER=   6.219 ms  speedup= 56.47x
N= 8  batch=  10000 | custom=  0.639 ms  cuSOLVER=   9.317 ms  speedup= 14.59x
N=16  batch=  10000 | custom=  5.259 ms  cuSOLVER=  13.427 ms  speedup=  2.55x
N=32  batch=  10000 | custom= 28.977 ms  cuSOLVER=  27.460 ms  speedup=  0.95x

[大 batch]
N= 3  batch= 100000 | custom=  0.539 ms  cuSOLVER=  56.055 ms  speedup=103.95x
N= 4  batch= 100000 | custom=  1.040 ms  cuSOLVER=  61.738 ms  speedup= 59.38x
N= 8  batch= 100000 | custom=  6.509 ms  cuSOLVER=  95.898 ms  speedup= 14.73x
N=16  batch= 100000 | custom= 57.258 ms  cuSOLVER= 137.170 ms  speedup=  2.40x

[超大 batch]
N= 3  batch=1000000 | custom=  4.830 ms  cuSOLVER= 556.171 ms  speedup=115.14x
N= 4  batch=1000000 | custom= 11.216 ms  cuSOLVER= 616.946 ms  speedup= 55.00x
```

### 9.3 精度

所有测试用例的重构相对 Frobenius 误差均 ≤ `1e-5`，FP32 精度极限附近。

| N | 自定义核误差 | cuSOLVER 误差 |
|---|---|---|
| 3 | 2.2e-7 | 7.3e-7 |
| 8 | 1.2e-6 | 1.8e-6 |
| 16 | 3.1e-6 | 2.8e-6 |
| 32 | 7.8e-6 | 4.8e-6 |

---

## 10. 使用范例

### 10.1 单矩阵分解

```cpp
#include <cuda_runtime.h>

int main() {
    // 准备 3x3 矩阵 (列主序)
    float hA[9] = { 1, 2, 3,    // 第 0 列
                    4, 5, 6,    // 第 1 列
                    7, 8, 9 };  // 第 2 列

    float *dA, *dU, *dS, *dV;
    cudaMalloc(&dA, sizeof(float) * 9);
    cudaMalloc(&dU, sizeof(float) * 9);
    cudaMalloc(&dS, sizeof(float) * 3);
    cudaMalloc(&dV, sizeof(float) * 9);

    cudaMemcpy(dA, hA, sizeof(float) * 9, cudaMemcpyHostToDevice);

    svdJacobiSmallKernel<3><<<1, 32>>>(dA, dU, dS, dV, 1);
    cudaDeviceSynchronize();

    float hU[9], hS[3], hV[9];
    cudaMemcpy(hU, dU, sizeof(float) * 9, cudaMemcpyDeviceToHost);
    cudaMemcpy(hS, dS, sizeof(float) * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(hV, dV, sizeof(float) * 9, cudaMemcpyDeviceToHost);

    // hS 已降序: hS[0] >= hS[1] >= hS[2]
    // hU, hV 列主序
}
```

### 10.2 批量分解（ICP 协方差对齐）

```cpp
constexpr int N = 3;
const int batch = 50000;  // 五万对点云的协方差矩阵

float *dCov, *dU, *dS, *dV;
cudaMalloc(&dCov, sizeof(float) * N * N * batch);
cudaMalloc(&dU,   sizeof(float) * N * N * batch);
cudaMalloc(&dS,   sizeof(float) * N * batch);
cudaMalloc(&dV,   sizeof(float) * N * N * batch);

// ... 填充 dCov ...

svdJacobiSmallKernel<N><<<batch, 32>>>(dCov, dU, dS, dV, batch);

// R = U * V^T 即为 ICP 的最优旋转矩阵
// 后续可在另一个 kernel 中拼装 R
```

### 10.3 跑基准

```cpp
int main() {
    benchVsCusolver<8>(10000, 42);
}
```

---

## 11. 适用边界与已知限制

### 11.1 适用范围

✅ N ∈ [2, 32]，**方阵**
✅ batch ≥ 几百，越多越能跑满 GPU
✅ 输入数据数值范围合理（避免极端尺度差异）
✅ FP32 单精度

### 11.2 不适用 / 退化场景

❌ **非方阵**（m ≠ n）— 当前 kernel 只处理方阵；非方阵请用 cuSOLVER 的 `gesvdj` / `gesvd`
❌ **N > 32** — 单 warp 装不下，需重写为 multi-warp
❌ **FP64 双精度** — 当前仅 `float`；改 `double` 需要复制一份模板（也会损失约 2× 速度）
❌ **极端病态矩阵**（条件数 > 1e6） — Jacobi 收敛慢，且 `MAX_SWEEPS=12` 可能不够；建议预先归一化或增加 sweep 数
❌ **batch < ~100** — GPU 利用率不足，与 cuSOLVER 的优势缩小

### 11.3 数值警告

- **零矩阵 / 零列**：对应奇异值为 0，`U` 的对应列被置零（不保留正交性）。重构误差仍为 0，但 `U` 不再是完全的正交矩阵
- **完全相同的奇异值**（重根）：对应的奇异向量在子空间内是任意旋转，不同算法可能给出不同结果。**这不是 bug，是 SVD 的本质性多解**
- **符号约定**：奇异值始终为正；`U`、`V` 的列符号由算法内部决定（FP32 数值上几乎不可能精确再现 LAPACK / NumPy 的符号）

### 11.4 线程安全

`svdJacobiSmallKernel` 本身是 device function，可以在多个 CUDA stream 上并发启动（输入输出不重叠的话）。`benchVsCusolver` 内部使用了非线程安全的 cusolverHandle 创建/销毁，**不要在多线程 host 代码中并发调用同一份**。

---

## 12. FAQ

### Q: 为什么 V 输出的是 V 不是 V^T？

A: 与 cuSOLVER `gesvdj` 一致，也方便上层做 `R = U * V^T`（直接矩阵乘）。如果需要 V^T，转置一次即可，开销 O(N²)。

### Q: `MAX_SWEEPS=12` 够用吗？

A: 对元素范围 [-1, 1] 的随机矩阵足够（实测收敛在 5-8 个 sweep）。对病态矩阵或大动态范围数据建议手动调高，最差不超过 30。

### Q: 能否对接 PyTorch / NumPy？

A: 可以。从 PyTorch `torch.Tensor.data_ptr()` 拿 device pointer，确保 `tensor.is_contiguous()` 且 `dtype=torch.float32` 后直接传入。注意 PyTorch 默认 **行主序**，需要 `tensor.t().contiguous()` 转列主序再调用。

### Q: N=32 为什么持平不再赢？

A: 单 warp 32 个线程在 N=32 时已全部用上，没有进一步并行余量；同时 cuSOLVER 在 N=32 这个公开记录的边界点上有专门优化。继续提速需要切换到 multi-warp + bank-conflict free 布局，工程量大，且 N=32 已经不是「典型小矩阵」了，建议直接转 cuSOLVER。

### Q: 为什么不用 closed-form 3×3 SVD（McAdams 2011）？

A: 算法值得做，对 N=3 可再提速 2-3×（彻底消除迭代）。当前 Jacobi 实现已达 ICP/SLAM 实际瓶颈（4.8ms / 100万次），暂未跟进。如有需要可作为后续优化。

### Q: 输出的 U, V 一定是正交矩阵吗？

A: 数值精度内是的。零奇异值情况下对应的 U 列会被置零（详见 11.3），此时 U 失去严格正交性，但重构 `U·diag(S)·V^T` 仍等于 A。

---

**作者**: assistant
**版本**: 2026-05-22
**许可**: 与所属项目一致
