#include <cuda_runtime.h>
#include <iostream>
#include "../include/CudaPractice.h"

__global__ void compute2DIndex(int *matrix, int width, int height)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    //确保索引不越界,过滤掉超额启动的线程
    if (row < height && col < width)
    {
        int index = row * width + col;
        matrix[index] = index;
    }
}

void printMatrix(int *matrix, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            std::cout << matrix[i * width + j] << "\t";
        }
        std::cout << std::endl;
    }
}

int IndexCalculation()
{
    int width = 9;
    int height = 8;
    //分配主机内存
    int *hostMatrix = new int[width * height];
    //分配设备内存
    int *deviceMatrix;
    cudaMalloc(&deviceMatrix, width * height* sizeof(int));
    //配置网格网络
    dim3 blockDim(4,4);          //每个线程块含4*4个线程
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,         //向上取整，确保每个分块都有线程处理
                (height + blockDim.y - 1) / blockDim.y);
    std::cout << "线程块维度：("<<blockDim.x<<","<<blockDim.y<<")"<<std::endl;
    std::cout << "网格维度:("<<gridDim.x<<","<<gridDim.y<<")"<<std::endl;

    //启动核函数
    compute2DIndex<<<gridDim,blockDim>>>(deviceMatrix, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy( hostMatrix,deviceMatrix, sizeof(int) * width * height,cudaMemcpyDeviceToHost);

    //打印结果
    std::cout<<"矩阵索引计算结果:"<<std::endl;
    printMatrix(hostMatrix, width, height);
    //释放内存
    delete[] hostMatrix;
    cudaFree(deviceMatrix);
    return 0;
}