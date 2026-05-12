#include <iostream>
#include "include/CudaPractice.h"

int print(const char *input)
{
    std::cout << input << std::endl;
    std::cout << std::endl;
    return 0;
}

int main()
{
    print("选择要演示的CUDA EXAMPLE的序号");
    print("1 : CUDA的网格与分块机制");
    print("2 : 寄存器溢出演示");
    print("3 : warp分支发散性能损失演示");
    print("4 : 基于矩阵加法的共享内存与全局内存比较");
    print("5 : 矩阵转置算法");
    print("6 : 通用矩阵乘法");
    print("7 : 方阵矩阵乘法");
    char choice = std::cin.get();

    switch (choice){
        case '1':
            IndexCalculation();
        case '2':
            RegisterOptimization();
        case '3':
            warpOptimization();
        case '4':
            MatrixAdd();
        case '5':
            transpose(4096,4096,4);
        case '6':
           sgemm(1024,1024,1024,32,4);
        case '7':
            squareMultiply(1024);
    }

    /*
    for (int i=0;i<4;++i)
    {
        sgemm(1024,1024,1024,32,4);
    }*/


}