#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "../include/CudaPractice.h"
#include <chrono>

// 核函数：使用共享内存进行矩阵转置
__global__ void blockMatrixTranspose(float *input, float *output, int N, int M)
{
    __shared__ float tile[32][32+1]; // +1 避免 bank conflict

    int xIndex = blockIdx.x * blockDim.x + threadIdx.x; // 原矩阵列
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y; // 原矩阵行

    // 将数据读入共享内存
    if (xIndex < M && yIndex < N)
        tile[threadIdx.y][threadIdx.x] = input[yIndex * M + xIndex];

    __syncthreads();

    // 转置后输出索引
    int transposed_x = blockIdx.y * blockDim.y + threadIdx.x;
    int transposed_y = blockIdx.x * blockDim.x + threadIdx.y;

    if (transposed_x < N && transposed_y < M)
        output[transposed_y * N + transposed_x] = tile[threadIdx.x][threadIdx.y];
}

int transpose(const int N, const int M, const int cudaStream_num)
{
    const size_t bytes = N * M * sizeof(float);

    // 主机内存分配并初始化
    float *hostInput;
    float *hostOutput;
    CUDA_CHECK(cudaHostAlloc(&hostInput, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hostOutput, bytes, cudaHostAllocDefault));

    for (int i = 0; i < N * M; i++)
        hostInput[i] = static_cast<float>(i);

    // 设备内存分配
    float *deviceInput, *deviceOutput;
    CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));

    auto start = std::chrono::high_resolution_clock::now();
    // 创建 CUDA 流
    std::vector<cudaStream_t> streams(cudaStream_num);
    for (int i = 0; i < cudaStream_num; i++)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    // 分块异步传输
    size_t chunk_size = (N * M) / cudaStream_num;
    for (int i = 0; i < cudaStream_num; i++)
    {
        size_t offset = i * chunk_size;
        size_t size = (i == cudaStream_num - 1) ? (N*M - offset) * sizeof(float) : chunk_size * sizeof(float);

        CUDA_CHECK(cudaMemcpyAsync(deviceInput + offset,
                                   hostInput + offset,
                                   size,
                                   cudaMemcpyHostToDevice,
                                   streams[i]));
    }

    // 等待所有数据传输完成
    for (int i = 0; i < cudaStream_num; i++)
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));

    // 设置线程块和网格
    dim3 dimBlock(32, 32);
    dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x,
                 (N + dimBlock.y - 1) / dimBlock.y);

    // 执行核函数
    blockMatrixTranspose<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, N, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 数据传回主机
    CUDA_CHECK(cudaMemcpy(hostOutput, deviceOutput, bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout<< "矩阵转置执行时间: " << duration.count() << " us" << std::endl;
    // 验证结果
    bool correct = true;
    for (int i = 0; i < N && correct; i++)
    {
        for (int j = 0; j < M; j++)
        {
            float original = hostInput[i * M + j];
            float transposed = hostOutput[j * N + i];
            if (abs(original - transposed) > 1e-5)
            {
                printf("Error at original[%d][%d]=%f, transposed[%d][%d]=%f\n",
                       i, j, original, j, i, transposed);
                correct = false;
                break;
            }
        }
    }

    if (correct)
        printf("Matrix transpose (N=%d, M=%d) successful!\n", N, M);

    // 清理内存
    for (int i = 0; i < cudaStream_num; i++)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    CUDA_CHECK(cudaFreeHost(hostInput));
    CUDA_CHECK(cudaFreeHost(hostOutput));

    return 0;
}