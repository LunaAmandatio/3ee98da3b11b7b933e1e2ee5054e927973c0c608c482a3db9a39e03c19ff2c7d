#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "../include/CudaPractice.h"

// 未优化：使用局部数组，寄存器溢出到本地内存
__global__ void registerOverflow(int* data, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        // 使用 volatile 防止优化，数组足够大导致寄存器溢出
        volatile int temp[256];  // 256 * 4 = 1024 字节
        #pragma unroll
        for (int i = 0; i < 256; i++)
        {
            temp[i] = idx * i;
        }

        int sum = 0;
        #pragma unroll
        for (int i = 0; i < 256; i++)
        {
            sum += temp[i];
        }
        data[idx] = sum;
    }
}

// 优化：直接计算，无临时数组
__global__ void optimizedRegisterUsage(int* data, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        int sum = 0;
        #pragma unroll
        for (int i = 0; i < 256; i++)
        {
            sum += idx * i;
        }
        data[idx] = sum;
    }
}


int RegisterOptimization()
{
    const int N = 1 << 21;  // 1M
    int *deviceData;

    cudaMalloc(&deviceData, N * sizeof(int));

    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    // 预热
    registerOverflow<<<gridDim, blockDim>>>(deviceData, N);
    cudaDeviceSynchronize();

    // 测试未优化版本
    auto start = std::chrono::high_resolution_clock::now();
    registerOverflow<<<gridDim, blockDim>>>(deviceData, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto overflowTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 测试优化版本
    start = std::chrono::high_resolution_clock::now();
    optimizedRegisterUsage<<<gridDim, blockDim>>>(deviceData, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto optimizedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "寄存器溢出版本: " << overflowTime.count() << " us" << std::endl;
    std::cout << "优化版本:       " << optimizedTime.count() << " us" << std::endl;

    cudaFree(deviceData);
    return 0;
}