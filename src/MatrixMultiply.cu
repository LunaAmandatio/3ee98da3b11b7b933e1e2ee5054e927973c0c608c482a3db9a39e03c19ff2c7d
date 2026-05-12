#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include "../include/CudaPractice.h"

#define CUDA_CHECK(err) if(err != cudaSuccess){ \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; exit(-1);}

// 核函数：共享内存 + B tile 转置
__global__ void matrixMultiplyTile(const float* A, const float* B, float* C, int M, int N, int K)
{
    const int TILE = 32;
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for(int t=0; t<(N+TILE-1)/TILE; t++)
    {
        //加载矩阵A到共享内存
        if(row < M && t*TILE + threadIdx.x < N)
        {
            tileA[threadIdx.y][threadIdx.x] = A[row*N + t*TILE + threadIdx.x];
        }
        else
        {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }


        //加载矩阵A到共享内存同时转置
        if(col < K && t*TILE + threadIdx.y < N)
            tileB[threadIdx.x][threadIdx.y] = B[(t*TILE + threadIdx.y)*K + col];
        else
            tileB[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        //求和累加
        #pragma unroll 32
        for(int i=0;i<TILE;i++)
        {
            acc += tileA[threadIdx.y][i]*tileB[threadIdx.x][i];
        }

        __syncthreads();
    }

    if(row < M && col < K)
        C[row*K + col] = acc;
}


// 初始化矩阵
void initializeMatrix(float* mat, int rows, int cols)
{
    for(int i=0;i<rows*cols;i++)
        mat[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
}


// 验证结果
bool verifyResult(float *hostC, float *hostA, float *hostB, int M, int N, int K)
{
    bool isCorrect = true;
    float epsilon = 1e-2f;

    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {
            float sum = 0.0f;
            for(int k=0;k<K;k++)
                sum += hostA[i*K + k] * hostB[k*N + j];
            if(fabs(hostC[i*N+j] - sum) > epsilon)
            {
                isCorrect = false;
                if(i<5 && j<5)
                    std::cout << "C["<<i<<"]["<<j<<"] GPU="<<hostC[i*N+j]<<", CPU="<<sum<<std::endl;
            }
        }
    }
    return isCorrect;
}


// 计算 FLOPS
void calculateFLOPS(int M, int N, int K, float elapsedTimeMs)
{
    double operations = 2.0 * M * N * K;  // 乘加次数
    double elapsedTimeSec = elapsedTimeMs / 1000.0;
    double flops = operations / elapsedTimeSec;

    std::cout << "\n========== FLOPS 性能统计 ==========" << std::endl;
    std::cout << "矩阵大小: (" << M << "x" << K << ") * (" << K << "x" << N << ") = (" << M << "x" << N << ")" << std::endl;
    std::cout << "总浮点操作数: " << operations << std::endl;
    std::cout << "计算时间: " << elapsedTimeMs << " ms (" << elapsedTimeSec << " s)" << std::endl;
    std::cout << "性能: " << flops/1e9 << " GFLOPS" << std::endl;
    std::cout << "=====================================" << std::endl;
}


// 主函数
int sgemm(int M, int N, int K, int BLOCK_SIZE=32, int STREAM_COUNT=4)
{
    size_t bytesA = M*N*sizeof(float);
    size_t bytesB = N*K*sizeof(float);
    size_t bytesC = M*K*sizeof(float);

    // 分配主机内存
    float *hA, *hB, *hC;
    CUDA_CHECK(cudaHostAlloc(&hA, bytesA, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hB, bytesB, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hC, bytesC, cudaHostAllocDefault));

    initializeMatrix(hA,M,N);
    initializeMatrix(hB,N,K);

    // 分配设备内存
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    // 创建 CUDA 流
    std::vector<cudaStream_t> streams(STREAM_COUNT);
    for(int i=0;i<STREAM_COUNT;i++)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    // 分块异步拷贝 A
    int rowsPerStream = (M + STREAM_COUNT - 1)/STREAM_COUNT;
    for(int i=0;i<STREAM_COUNT;i++)
    {
        int rowStart = i*rowsPerStream;
        int rowCount = std::min(rowsPerStream, M - rowStart);
        if(rowCount>0)
        {
            CUDA_CHECK(cudaMemcpyAsync(dA + rowStart*N,
                hA + rowStart*N,
                rowCount*N*sizeof(float),
                cudaMemcpyHostToDevice,
                streams[i]));
        }
    }

    // B 一次性拷贝
    CUDA_CHECK(cudaMemcpy(dB,hB,bytesB,cudaMemcpyHostToDevice));

    // 等待数据拷贝完成
    for(int i=0;i<STREAM_COUNT;i++)
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));

    // 配置网格
    dim3 block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid((K+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);

    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    // 启动核函数
    matrixMultiplyTile<<<grid,block>>>(dA,dB,dC,M,N,K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷贝结果回 host
    CUDA_CHECK(cudaMemcpy(hC,dC,bytesC,cudaMemcpyDeviceToHost));

    //计时结束
    auto end = std::chrono::high_resolution_clock::now();
    float elapsedMs = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/1000.0f;

    // 验证
    if(verifyResult(hC,hA,hB,M,N,K))
        std::cout << "Verification passed!" << std::endl;
    else
        std::cout << "Verification failed!" << std::endl;

    // 计算性能
    calculateFLOPS(M,N,K,elapsedMs);

    // 释放内存
    for(int i=0;i<STREAM_COUNT;i++)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFreeHost(hA));
    CUDA_CHECK(cudaFreeHost(hB));
    CUDA_CHECK(cudaFreeHost(hC));

    return 0;
}
