#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

using namespace std;

//核函数,使用Warp Shuffle实现warp内规约求和
__global__ void warpReduceSum(int *input,unsigned long long *output,int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;      //线程在warp内的索引
    int warpId = threadIdx.x / warpSize;
    //初始化规约值
    long long value=(idx<N) ? input[idx] : 0;
    //使用Warp Shuffle实现规约
    for (int offset=warpSize/2;offset>0;offset /= 2)
    {
        value += __shfl_down_sync(0xffffffff,value,offset);
    }
    //将每个Warp的结果写入共享内存
    if (lane==0)
    {
        atomicAdd(output,value);        //原子操作，避免冲突
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

int main()
{
    const int dataSize=1 << 20;
    int *hostInput= new int[dataSize];
    unsigned long long hostOutput= 0;
    //初始化输入数据
    for (int i=0; i<dataSize; i++)
    {
        hostInput[i]=i;
    }
    //分配设备内存
    int *deviceInput ;
    unsigned long long *deviceOutput;
    cudaMalloc(&deviceInput,dataSize*sizeof(int));
    cudaMalloc(&deviceOutput,sizeof(unsigned long long));
    checkCudaError("Device memory allocation failed");
    //复制数据到设备
    cudaMemcpy(deviceInput, hostInput, dataSize*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(deviceOutput,0,sizeof(unsigned long long));     //初始化输出为0
    checkCudaError("Data copy failed");
    //配置线程块和网格
    dim3 blockDim(256);
    dim3 gridDim((dataSize + blockDim.x - 1) / blockDim.x);
    //启动核函数
    auto start = chrono::high_resolution_clock::now();
    warpReduceSum<<<gridDim,blockDim>>>(deviceInput,deviceOutput,dataSize);
    cudaDeviceSynchronize();
    checkCudaError("kernel execution failed");
    auto end = chrono::high_resolution_clock::now();
    //复制结果回host
    cudaMemcpy(&hostOutput, deviceOutput, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    checkCudaError("Data copy failed");
    //输出结果
    std::cout<<"数组总和:"<<hostOutput<<std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout<<"执行时间:" <<duration.count()<<"us"<<std::endl;
    //释放内存
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    return 0;
}