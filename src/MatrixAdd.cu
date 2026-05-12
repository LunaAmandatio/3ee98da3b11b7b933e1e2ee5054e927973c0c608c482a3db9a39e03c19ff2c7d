#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "../include/CudaPractice.h"

//矩阵加法核函数,使用全局内存
__global__ void matrixAddGlobalMemory(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.x + threadIdx.y;
    if (idx < N && idy < N)
    {
        int index = idy * N + idx;
        C[index] = A[index] + B[index];
    }
}

//矩阵加法核函数,使用共享内存
__global__ void matrixAddSharedMemory(const float *A,const float *B, float *C, int N)
{
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];
    int tx =threadIdx.x, ty = threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.x + threadIdx.y;
    if (idx < N && idy < N)
    {
        //将全局内存数据加载到共享内存
        int index = idy * N + idx;
        tileA[ty][tx] = A[index];
        tileB[ty][tx] = B[index];
        __syncthreads();
        //执行矩阵加法
        C[index] = tileA[ty][tx] + tileB[ty][tx];
    }
}

//atomicAdd 函数需要修改内存中的值，所以参数类型必须是非const指针
__global__ void matrixAddAtomic(float *input,float *result,int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        atomicAdd(result,input[idx]);
    }
}

//检查cuda错误
void checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA ERROR: " << msg << std::endl;
        exit(EXIT_FAILURE);
    }
}

int MatrixAdd()
{
    const int N = 1024;
    const int size = N*N;
    float *hostA = new float[size];
    float *hostB = new float[size];
    float *hostC = new float[size];
    //初始化矩阵数据
    for (int i = 0; i < size; i++)
    {
        hostA[i] = 1.0f;
        hostB[i] = 2.0f;
    }
    // 分配设备内存
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, sizeof(float)*size);
    cudaMalloc(&deviceB, sizeof(float)*size);
    cudaMalloc(&deviceC, sizeof(float)*size);
    checkCudaError("Device memory allocation failed");

    //复制数据到device
    cudaMemcpy(deviceA, hostA, sizeof(float)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(float)*size, cudaMemcpyHostToDevice);
    checkCudaError("Data copy from host to device failed");

    //配置线程块和网格
    dim3 blockDim(32,32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
             (N + blockDim.y - 1) / blockDim.y);

    //全局内存版本计时
    auto start = std::chrono::high_resolution_clock::now();
    matrixAddGlobalMemory<<<gridDim,blockDim>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto globalDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "全局内存版本执行时间: " << globalDuration.count() << " us" << std::endl;

    //共享内存版本计时
    start = std::chrono::high_resolution_clock::now();
    matrixAddSharedMemory<<<gridDim,blockDim>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto sharedDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout<< "共享内存版本执行时间: " << sharedDuration.count() << " us" << std::endl;

    std::cout<< "共享内存版本加速比:"<< globalDuration.count()/sharedDuration.count() <<std::endl;

    //清理缓存
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
    return 0;
}