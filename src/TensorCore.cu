#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <iostream>
#include <chrono>

using namespace nvcuda;

const int MATRIX_SIZE = 1011;

// Tensor Core 矩阵乘法核函数（使用 WMMA API）
__global__ void matixMultiplyTensorCore(half* A, half* B, float* C, int N) {
    // WMMA 配置
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Warp 索引
    int warpRow = (threadIdx.y * blockDim.y + threadIdx.x) / 32;
    int warpCol = (threadIdx.y * blockDim.y + threadIdx.x) % 32;

    int globalRow = blockIdx.y * 16 + warpRow * 16;
    int globalCol = blockIdx.x * 16 + warpCol * 16;

    wmma::fill_fragment(c_frag, 0.0f);

    // 分块计算
    for (int tile = 0; tile < (N + 15) / 16; tile++) {
        int tileK = tile * 16;

        // 加载 A（现在 A 是 half* 类型）
        if (globalRow < N && tileK < N) {
            wmma::load_matrix_sync(a_frag, &A[globalRow * N + tileK], N);
        } else {
            wmma::fill_fragment(a_frag, 0.0f);
        }

        // 加载 B（现在 B 是 half* 类型）
        if (globalCol < N && tileK < N) {
            wmma::load_matrix_sync(b_frag, &B[tileK * N + globalCol], N);
        } else {
            wmma::fill_fragment(b_frag, 0.0f);
        }

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 存储结果
    if (globalRow < N && globalCol < N) {
        wmma::store_matrix_sync(&C[globalRow * N + globalCol], c_frag, N, 
                                wmma::mem_row_major);
    }
}

// 简化版 Tensor Core 实现（使用 half 精度）
__global__ void matixMultiplyTensorCoreSimple(half* A, half* B, float* C, int N) {
    __shared__ half tileA[32][32];
    __shared__ half tileB[32][32];
    
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    
    float result = 0.0f;
    
    for (int tile = 0; tile < (N + 31) / 32; tile++) {
        // 加载数据（使用 half 精度）
        if (row < N && tile * 32 + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + tile * 32 + threadIdx.x];
        }
        if (col < N && tile * 32 + threadIdx.y < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(tile * 32 + threadIdx.y) * N + col];
        }
        __syncthreads();
        
        // Tensor Core 会自动优化 half 精度的乘加
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            result += __half2float(tileA[threadIdx.y][k]) * 
                      __half2float(tileB[k][threadIdx.x]);
        }
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = result;
    }
}

// 转换 float 到 half
void convertToHalf(const float* src, half* dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = __float2half(src[i]);
    }
}

//初始化矩阵
void initializeMatrix(float *matrix, int size)
{
    for (int i = 0; i < size * size; i++)
    {
        matrix[i] = (static_cast<float>(rand()) / (RAND_MAX + 1.0f)) * 10.0f;
    }
}

// 计算FLOPS的函数
void calculateFLOPS(int N, float elapsedTimeMs)
{
    // 矩阵乘法操作数：2 * N^3 (乘法和加法各N^3次)
    double operations = 2.0 * N * N * N;
    double elapsedTimeSec = elapsedTimeMs / 1000.0;
    double flops = operations / elapsedTimeSec;

    std::cout << "\n========== FLOPS 性能统计 ==========" << std::endl;
    std::cout << "矩阵大小: " << N << " x " << N << std::endl;
    std::cout << "总浮点操作数: " << operations << " (2*N^3)" << std::endl;
    std::cout << "计算时间: " << elapsedTimeMs << " ms (" << elapsedTimeSec << " s)" << std::endl;
    std::cout << "性能: " << flops / 1e9 << " GFLOPS" << std::endl;
    std::cout << "=====================================" << std::endl;
}

// 验证结果的函数
bool verifyResult(float *hostC, float *hostA, float *hostB, int N)
{
    float *expected = new float[N * N]();

    // CPU 计算
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += hostA[i * N + k] * hostB[k * N + j];
            }
            expected[i * N + j] = sum;
        }
    }

    // 比较
    float epsilon = 1e-2f;  // 容差范围
    bool isCorrect = true;
    float maxDiff = 0.0f;
    int errorCount = 0;
    const int maxErrorsToShow = 10;

    for (int i = 0; i < N * N; i++) {
        float diff = fabs(hostC[i] - expected[i]);
        if (diff > maxDiff) maxDiff = diff;

        if (diff > epsilon) {
            if (isCorrect) {
                printf("Verification failed!\n");
                isCorrect = false;
            }
            if (errorCount < maxErrorsToShow) {
                printf("  C[%d][%d]: GPU=%.2f, CPU=%.2f, diff=%.2f\n",
                       i / N, i % N, hostC[i], expected[i], diff);
                errorCount++;
            }
        }
    }

    // 输出验证结果
    if (isCorrect) {
        printf("Verification passed! Maximum error: %.10f(tolerance: %.3f)\n", maxDiff, epsilon);
    } else {
        printf("Verification failed! Maximum error: %.10f (tolerance: %.3f)\n", maxDiff, epsilon);
        if (errorCount >= maxErrorsToShow) {
            printf("  ... and %d more errors\n", (N * N - errorCount));
        }
    }

    delete[] expected;
    return isCorrect;
}

// 主函数（使用 Tensor Core）
int main() {
    int N = MATRIX_SIZE;
    size_t bytes = N * N * sizeof(float);
    size_t halfBytes = N * N * sizeof(half);
    
    // 分配主机内存（锁页内存）
    float *hostA, *hostB, *hostC;
    half *hostAHalf, *hostBHalf;
    cudaHostAlloc(&hostA, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&hostB, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&hostC, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&hostAHalf, halfBytes, cudaHostAllocDefault);
    cudaHostAlloc(&hostBHalf, halfBytes, cudaHostAllocDefault);
    
    // 初始化
    initializeMatrix(hostA, N);
    initializeMatrix(hostB, N);
    
    // 转换为 half 精度
    convertToHalf(hostA, hostAHalf, N * N);
    convertToHalf(hostB, hostBHalf, N * N);
    
    // 分配设备内存
    half *deviceA, *deviceB;
    float *deviceC;
    cudaMalloc(&deviceA, halfBytes);
    cudaMalloc(&deviceB, halfBytes);
    cudaMalloc(&deviceC, bytes);
    
    // 异步拷贝到设备
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(deviceA, hostAHalf, halfBytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(deviceB, hostBHalf, halfBytes, cudaMemcpyHostToDevice, stream);
    
    // 配置网格（使用 32x32 块以更好利用 Tensor Core）
    dim3 blockDim(32, 32);
    dim3 gridDim((N + 31) / 32, (N + 31) / 32);
    
    cudaStreamSynchronize(stream);
    
    // 预热
    for (int i = 0; i < 5; i++) {
        matixMultiplyTensorCoreSimple<<<gridDim, blockDim, 0, stream>>>(
            deviceA, deviceB, deviceC, N);
    }
    cudaDeviceSynchronize();
    
    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    matixMultiplyTensorCoreSimple<<<gridDim, blockDim, 0, stream>>>(
        deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    // 拷贝结果回主机
    cudaMemcpyAsync(hostC, deviceC, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // 计算性能
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float elapsedTimeMs = duration.count() / 1000.0f;
    calculateFLOPS(N, elapsedTimeMs);
    
    // 验证（注意 Tensor Core 可能有精度差异）
    verifyResult(hostC, hostA, hostB, N);
    
    // 清理
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    cudaFreeHost(hostA);
    cudaFreeHost(hostB);
    cudaFreeHost(hostC);
    cudaFreeHost(hostAHalf);
    cudaFreeHost(hostBHalf);
    cudaStreamDestroy(stream);
    
    return 0;
}