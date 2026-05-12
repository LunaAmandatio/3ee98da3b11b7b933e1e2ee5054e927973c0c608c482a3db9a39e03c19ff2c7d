#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include "../include/CudaPractice.h"

//核函数，基于共享内存的矩阵乘法
__global__ void matixMultiplyShared(float* A, float* B, float* C, int N)
{
    //共享内存分配
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];
    //当前线程的行和列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    //分块加载A,B计算
    for (int tile=0; tile<(N+15)/16; tile++)      //依旧取整算法
    {
        //加载A的分块到共享内存
        if (row<N && tile*16+threadIdx.x<N)
        {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + tile * 16 + threadIdx.x];
        }
        else
        {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        //加载B的分块到共享内存
        if (col<N && tile*16+threadIdx.y<N)
        {
            tileB[threadIdx.y][threadIdx.x] = B[(tile*16 + threadIdx.y)*N + col];
        }
        else
        {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();              //确保所有线程加载完成
        //计算C[row][col]
        for (int k=0;k<16;++k)
        {
            result += tileA[threadIdx.y][k]*tileB[k][threadIdx.x];
        }
        __syncthreads();              //确保所有线程计算完成
    }
    //将结果写入C矩阵
    if (row < N && col < N)
    {
        C[row*N + col] = result;
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
//打印矩阵
void printMatrix(float *matrix,int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            std::cout << matrix[i*size+j] << " \t";
        }
        std::cout << std::endl;
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

void Warmup(float *deviceA, float *deviceB, float *deviceC,
                    int N, dim3 gridDim, dim3 blockDim, int warmupIterations = 5)
{
    std::cout << "Performing " << warmupIterations << " warmup iterations..." << std::endl;

    for (int i = 0; i < warmupIterations; i++)
    {
        matixMultiplyShared<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, N);
        cudaDeviceSynchronize();

        if (i == 0 || i == warmupIterations - 1)
            std::cout << "  Warmup " << (i + 1) << "/" << warmupIterations << " finished" << std::endl;
        else if (i == 1)
            std::cout << "  ..." << std::endl;
    }

    std::cout << "Warmup completed!" << std::endl;
}

int squareMultiply(int N = 1024)
{
    //分配主机内存
    float *hostA;
    float *hostB;
    float *hostC;
    cudaHostAlloc(&hostA, N * N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&hostB, N * N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&hostC, N * N * sizeof(float), cudaHostAllocDefault);

    //初始化矩阵A,B
    initializeMatrix(hostA,N);
    initializeMatrix(hostB,N);

    //分配设备内存
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, N*N*sizeof(float));
    cudaMalloc(&deviceB, N*N*sizeof(float));
    cudaMalloc(&deviceC, N*N*sizeof(float));

    //复制数据到设备
    //cudaMemcpy(deviceA, hostA, N*N*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(deviceB, hostB, N*N*sizeof(float), cudaMemcpyHostToDevice);

    //cuda流创建
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    //异步拷贝
    cudaMemcpyAsync(deviceA, hostA, N*N*sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(deviceB, hostB, N*N*sizeof(float), cudaMemcpyHostToDevice, stream2);

    //配置网格网络
    dim3 blockDim(16,16);
    dim3 gridDim((N+blockDim.x-1)/blockDim.x,(N+blockDim.y-1)/blockDim.y);

    // 调用预热函数
    Warmup(deviceA, deviceB, deviceC, N, gridDim, blockDim, 15);

    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();

    //核函数启动
    matixMultiplyShared<<<gridDim,blockDim,0,stream1>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();

    //复制结果回host
    //cudaMemcpy(hostC,deviceC, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(hostC, deviceC, N*N*sizeof(float), cudaMemcpyDeviceToHost, stream1);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float elapsedTimeMs = duration.count() / 1000.0f;
    //打印矩阵
    /*
    std::cout <<"矩阵A:" <<std::endl;
    printMatrix(hostA,N);
    std::cout <<"矩阵B:" <<std::endl;
    printMatrix(hostB,N);
    std::cout <<"矩阵C(结果):" <<std::endl;
    printMatrix(hostC,N);
    */

    // 计算并打印FLOPS
    calculateFLOPS(N, elapsedTimeMs);

    // 验证结果
    if (verifyResult(hostC, hostA, hostB, N)) {
        std::cout << "True" << std::endl;
    } else {
        std::cout << "Wrong" << std::endl;
    }

    //释放内存
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    cudaFreeHost(hostA);
    cudaFreeHost(hostB);
    cudaFreeHost(hostC);
    return 0;
}