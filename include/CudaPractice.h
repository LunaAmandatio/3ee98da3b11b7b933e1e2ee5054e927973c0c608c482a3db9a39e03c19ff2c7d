#ifndef DGEMM_CUDAPRACTICE_H
#define DGEMM_CUDAPRACTICE_H

#endif //DGEMM_CUDAPRACTICE_H


//检查cuda错误
#define CUDA_CHECK(call) \
{ \
cudaError_t err = call; \
if (err != cudaSuccess){ \
std::cerr << "CUDA ERROR at " << __FILE__ <<":"<<__LINE__<<":" <<cudaGetErrorString(err)<<std::endl; \
exit(EXIT_FAILURE); \
} \
}

//CUDA的网格划分演示
int IndexCalculation();

//寄存器溢出损失演示
int RegisterOptimization();

//warp分支发散性能损失演示
int warpOptimization();

//基于矩阵加法的共享内存与全局内存比较
int MatrixAdd();

//矩阵转置算法
int transpose(const int N,const int M,const int cudaStream_num);

//矩阵乘法
// N 为矩阵A行数
// M 为 矩阵A列数 与 矩阵B行数
// K 为 矩阵B列数
//BLOCK_SIZE 为 线程块大小
//BLOCK_COUNT 为 分块数量
int sgemm(int M,int N,int K,int BLOCK_SIZE,int BLOCK_COUNT);

//方阵矩阵乘法
int squareMultiply(int N);