#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ImageManager.hpp"

// CUDA kernel for pixel-wise addition with clamping
__global__ void addKernel(const unsigned char* a,
    const unsigned char* b,
    unsigned char* out,
    int width,
    int height,
    int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        for (int c = 0; c < channels; ++c) {
            int v = a[idx + c] + b[idx + c];
            out[idx + c] = static_cast<unsigned char>(v > 255 ? 255 : v);
        }
    }
}

// CUDA kernel for pixel-wise subtraction with clamping
__global__ void subKernel(const unsigned char* a,
    const unsigned char* b,
    unsigned char* out,
    int width,
    int height,
    int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        for (int c = 0; c < channels; ++c) {
            int v = a[idx + c] - b[idx + c];
            out[idx + c] = static_cast<unsigned char>(v < 0 ? 0 : v);
        }
    }
}

// Host wrapper
void launchAdd(const Image& imgA, const Image& imgB, Image& imgOut) {
    int size = imgA.width * imgA.height * imgA.channels;
    unsigned char* d_a, * d_b, * d_out;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_a, imgA.data.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, imgB.data.data(), size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((imgA.width + block.x - 1) / block.x,
        (imgA.height + block.y - 1) / block.y);
    addKernel << <grid, block >> > (d_a, d_b, d_out,
        imgA.width, imgA.height, imgA.channels);
    cudaDeviceSynchronize();

    cudaMemcpy(imgOut.data.data(), d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

void launchSub(const Image& imgA, const Image& imgB, Image& imgOut) {
    int size = imgA.width * imgA.height * imgA.channels;
    unsigned char* d_a, * d_b, * d_out;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_a, imgA.data.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, imgB.data.data(), size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((imgA.width + block.x - 1) / block.x,
        (imgA.height + block.y - 1) / block.y);
    subKernel << <grid, block >> > (d_a, d_b, d_out,
        imgA.width, imgA.height, imgA.channels);
    cudaDeviceSynchronize();

    cudaMemcpy(imgOut.data.data(), d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}