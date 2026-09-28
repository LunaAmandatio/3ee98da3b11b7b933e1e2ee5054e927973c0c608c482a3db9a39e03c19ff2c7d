#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define BLOCK_DIM 16
#define TILE_UNROLL 4

// FP16 RGB packed
struct half4 {
    half x, y, z, w; // w 保留不用
};

// 核函数：FP16 + Tile + Loop Unrolling + uchar4 packed
__global__ void resize_kernel_fp16_rgb(const uchar4* __restrict__ src, int src_w, int src_h,
                                       uchar4* dst, int dst_w, int dst_h) {
    __shared__ half4 tile[BLOCK_DIM + 1][BLOCK_DIM + 1];

    int x_out_base = blockIdx.x * BLOCK_DIM;
    int y_out_base = blockIdx.y * BLOCK_DIM;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int uy = 0; uy < TILE_UNROLL; ++uy) {
        for (int ux = 0; ux < TILE_UNROLL; ++ux) {
            int x_out = x_out_base + tx + ux;
            int y_out = y_out_base + ty + uy;
            if (x_out >= dst_w || y_out >= dst_h) continue;

            float scale_x = (float)src_w / dst_w;
            float scale_y = (float)src_h / dst_h;
            float src_x = x_out * scale_x;
            float src_y = y_out * scale_y;

            int x0 = floorf(src_x);
            int y0 = floorf(src_y);
            int x1 = min(x0 + 1, src_w - 1);
            int y1 = min(y0 + 1, src_h - 1);

            float dx = src_x - x0;
            float dy = src_y - y0;

            // 将输入图像加载到共享内存 (FP16)
            uchar4 p00 = src[y0 * src_w + x0];
            uchar4 p01 = src[y0 * src_w + x1];
            uchar4 p10 = src[y1 * src_w + x0];
            uchar4 p11 = src[y1 * src_w + x1];

            tile[ty][tx].x = __float2half(p00.x / 255.0f);
            tile[ty][tx].y = __float2half(p00.y / 255.0f);
            tile[ty][tx].z = __float2half(p00.z / 255.0f);

            tile[ty][tx + 1].x = __float2half(p01.x / 255.0f);
            tile[ty][tx + 1].y = __float2half(p01.y / 255.0f);
            tile[ty][tx + 1].z = __float2half(p01.z / 255.0f);

            tile[ty + 1][tx].x = __float2half(p10.x / 255.0f);
            tile[ty + 1][tx].y = __float2half(p10.y / 255.0f);
            tile[ty + 1][tx].z = __float2half(p10.z / 255.0f);

            tile[ty + 1][tx + 1].x = __float2half(p11.x / 255.0f);
            tile[ty + 1][tx + 1].y = __float2half(p11.y / 255.0f);
            tile[ty + 1][tx + 1].z = __float2half(p11.z / 255.0f);

            __syncthreads();

            // 双线性插值
            half4 res;
            res.x = __hadd(__hmul(__hsub(__float2half(1.0f), __float2half(dx)),
                                   __hmul(__hsub(__float2half(1.0f), __float2half(dy)), tile[ty][tx].x)),
                           __hmul(__float2half(dx), __hmul(__hsub(__float2half(1.0f), __float2half(dy)), tile[ty][tx + 1].x)));
            res.x = __hadd(res.x, __hmul(__hsub(__float2half(1.0f), __float2half(dx)),
                                         __hmul(__float2half(dy), tile[ty + 1][tx].x)));
            res.x = __hadd(res.x, __hmul(__float2half(dx), __hmul(__float2half(dy), tile[ty + 1][tx + 1].x)));

            res.y = __hadd(__hmul(__hsub(__float2half(1.0f), __float2half(dx)),
                                   __hmul(__hsub(__float2half(1.0f), __float2half(dy)), tile[ty][tx].y)),
                           __hmul(__float2half(dx), __hmul(__hsub(__float2half(1.0f), __float2half(dy)), tile[ty][tx + 1].y)));
            res.y = __hadd(res.y, __hmul(__hsub(__float2half(1.0f), __float2half(dx)),
                                         __hmul(__float2half(dy), tile[ty + 1][tx].y)));
            res.y = __hadd(res.y, __hmul(__float2half(dx), __hmul(__float2half(dy), tile[ty + 1][tx + 1].y)));

            res.z = __hadd(__hmul(__hsub(__float2half(1.0f), __float2half(dx)),
                                   __hmul(__hsub(__float2half(1.0f), __float2half(dy)), tile[ty][tx].z)),
                           __hmul(__float2half(dx), __hmul(__hsub(__float2half(1.0f), __float2half(dy)), tile[ty][tx + 1].z)));
            res.z = __hadd(res.z, __hmul(__hsub(__float2half(1.0f), __float2half(dx)),
                                         __hmul(__float2half(dy), tile[ty + 1][tx].z)));
            res.z = __hadd(res.z, __hmul(__float2half(dx), __hmul(__float2half(dy), tile[ty + 1][tx + 1].z)));

            // 存储回全局内存 uchar4
            uchar4 out_pixel;
            out_pixel.x = static_cast<uchar>(__half2float(res.x) * 255.0f);
            out_pixel.y = static_cast<uchar>(__half2float(res.y) * 255.0f);
            out_pixel.z = static_cast<uchar>(__half2float(res.z) * 255.0f);
            out_pixel.w = 0;
            dst[y_out * dst_w + x_out] = out_pixel;
        }
    }
}

// ------------------- 主函数 -------------------
int main() {
    cv::Mat img = cv::imread("D:/Project/drone-track/src/yolo_modules/opencv_inference/test.jpg");
    if (img.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    int src_w = img.cols;
    int src_h = img.rows;
    int dst_w = 640;
    int dst_h = 640;

    cv::Mat img_rgba;
    cv::cvtColor(img, img_rgba, cv::COLOR_BGR2BGRA);

    uchar4* d_src;
    uchar4* d_dst;
    size_t src_bytes = src_w * src_h * sizeof(uchar4);
    size_t dst_bytes = dst_w * dst_h * sizeof(uchar4);

    cudaMalloc(&d_src, src_bytes);
    cudaMalloc(&d_dst, dst_bytes);

    auto start_total = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_src, img_rgba.ptr<uchar4>(), src_bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((dst_w + BLOCK_DIM - 1) / BLOCK_DIM, (dst_h + BLOCK_DIM - 1) / BLOCK_DIM);

    auto start_kernel = std::chrono::high_resolution_clock::now();
    resize_kernel_fp16_rgb<<<grid, block>>>(d_src, src_w, src_h, d_dst, dst_w, dst_h);
    cudaDeviceSynchronize();
    auto end_kernel = std::chrono::high_resolution_clock::now();

    cv::Mat out(dst_h, dst_w, CV_8UC4);
    cudaMemcpy(out.ptr<uchar4>(), d_dst, dst_bytes, cudaMemcpyDeviceToHost);

    auto end_total = std::chrono::high_resolution_clock::now();

    cv::cvtColor(out, out, cv::COLOR_BGRA2BGR);
    cv::imwrite("output_2.0.jpg", out);

    cudaFree(d_src);
    cudaFree(d_dst);

    auto kernel_ms = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    auto total_ms  = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    std::cout << "Kernel execution time: " << kernel_ms << " ms\n";
    std::cout << "Total execution time: " << total_ms << " ms\n";
}