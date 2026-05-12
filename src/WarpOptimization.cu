#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "../include/CudaPractice.h"

//核函数，存在发散
__global__ void branchDivergence(int *data,int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        if (idx % 2 == 0)
        {
            data[idx] *=2;
        }
        else
        {
            data[idx] += 1;
        }
    }
}

//核函数，通过分支规约优化
__global__ void branchReduction(int *data,int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        int value = data[idx];
        data[idx] = (idx % 2 == 0) ? value * 2 : value+1;
    }
}



int warpOptimization()
{
    const int datasize = 1 << 20;       //数据量1MB
    int *hostData = new int[datasize];
    int *deviceData;
    //初始化host数据
    for (int i = 0; i < datasize; i++)
    {
        hostData[i] = i;
    }
    //分配设备内存
    cudaMalloc(&deviceData, sizeof(int) * datasize);
    //将数据从host拷贝到device
    cudaMemcpy(deviceData, hostData, datasize*sizeof(int), cudaMemcpyHostToDevice);
    //测试分支发散
    dim3 blockDim(256);
    dim3 gridDim((datasize + blockDim.x - 1) / blockDim.x);
    auto start = std::chrono::high_resolution_clock::now();
    branchDivergence<<<gridDim, blockDim>>>(deviceData, datasize);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout<<"Branch divergence kernel execution time: "<<duration.count()<<" us"<<std::endl;
    //测试分支规约
    start = std::chrono::high_resolution_clock::now();
    branchReduction<<<gridDim, blockDim>>>(deviceData, datasize);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout<<"Branch reduction kernel execution time: "<<duration.count()<<" us"<<std::endl;

    cudaFree(deviceData);
    delete[] hostData;
    return 0;
}