cmake_minimum_required(VERSION 3.20)
project(CudaPractice LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 120)

# 显式添加CUDA包含目录
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include")
find_package(CUDAToolkit REQUIRED)


# 设置 OpenBLAS 路径
set(OpenBLAS_PATH "D:/MinGW/mingw64" CACHE PATH "OpenBLAS installation directory")


# 查找 OpenBLAS
find_library(OpenBLAS_LIB
        NAMES openblas libopenblas libopenblas.dll.a openblas.lib
        PATHS ${OpenBLAS_PATH}/lib
        REQUIRED
)

if(NOT OpenBLAS_LIB)
    message(FATAL_ERROR "OpenBLAS library not found. Please install OpenBLAS.")
endif()

message(STATUS "Found OpenBLAS: ${OpenBLAS_LIB}")


# 查找 OpenMP
find_package(OpenMP REQUIRED)


#添加可执行文件
add_executable(CudaPractice
        main.cpp
        src/IndexCalculation.cu
        src/RegisterOverflow.cu
        src/WarpOptimization.cu
        src/MatrixAdd.cu
        src/Transpose.cu
        src/MatrixMultiply.cu
        src/MatrixMultiplyShared.cu
)

# 为不同语言设置不同的编译选项
target_compile_options(CudaPractice PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)

# OpenMP 链接
if(OpenMP_CXX_FOUND)
    target_link_libraries(CudaPractice PRIVATE OpenMP::OpenMP_CXX)
endif()

# CUDA 特定选项
target_compile_options(CudaPractice PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

# 链接 cuBLAS 和 CUDA 运行时库
target_link_libraries(CudaPractice PRIVATE
        CUDA::cudart
        CUDA::cublas
        ${OpenBLAS_LIB}
)