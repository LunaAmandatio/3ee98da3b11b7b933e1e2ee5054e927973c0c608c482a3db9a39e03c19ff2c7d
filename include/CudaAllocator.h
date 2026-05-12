#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "helper_cuda.h"

template <class T>
struct CudaAllocator {
    using value_type = T;

    // 必须的构造函数
    CudaAllocator() noexcept = default;
    CudaAllocator(const CudaAllocator&) noexcept = default;

    template <class U>
    CudaAllocator(const CudaAllocator<U>&) noexcept {}

    T *allocate(size_t size) {
        T *ptr = nullptr;
        checkCudaErrors(cudaMallocManaged(&ptr, size * sizeof(T)));
        return ptr;
    }

    void deallocate(T *ptr, size_t size = 0) {
        checkCudaErrors(cudaFree(ptr));
    }
};

// 必须的比较操作符
template <class T, class U>
bool operator==(const CudaAllocator<T>&, const CudaAllocator<U>&) noexcept {
    return true;
}

template <class T, class U>
bool operator!=(const CudaAllocator<T>&, const CudaAllocator<U>&) noexcept {
    return false;
}