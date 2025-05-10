#include "ImageManager.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

namespace GPU {

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

    __global__ void smoothKernel(const unsigned char* in, unsigned char* out,
        int w, int h, int ch)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= w || y >= h) return;

        int idx = (y * w + x) * ch;

        for (int c = 0;c < ch;++c) {
            int sum = 0, cnt = 0;
            for (int ky = -1;ky <= 1;++ky)
                for (int kx = -1;kx <= 1;++kx) {
                    int nx = x + kx, ny = y + ky;
                    if (nx >= 0 && ny >= 0 && nx < w && ny < h) {
                        int nidx = (ny * w + nx) * ch + c;
                        sum += in[nidx];
                        ++cnt;
                    }
                }
            out[idx + c] = static_cast<unsigned char>(sum / cnt);
        }
    }

    __global__ void enhanceKernel(const unsigned char* in,
        unsigned char* out,
        int w, int h, int ch)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= w || y >= h) return;

        int idx = (y * w + x) * ch;
        unsigned char blur[4] = { 0 };

        // 3x3 box blur inline
        for (int c = 0;c < ch;++c) {
            int sum = 0, cnt = 0;
            for (int ky = -1;ky <= 1;++ky)
                for (int kx = -1;kx <= 1;++kx) {
                    int nx = x + kx, ny = y + ky;
                    if (nx >= 0 && ny >= 0 && nx < w && ny < h) {
                        sum += in[(ny * w + nx) * ch + c]; ++cnt;
                    }
                }
            blur[c] = sum / cnt;
        }

        for (int c = 0;c < ch;++c) {
            int v = in[idx + c] + (in[idx + c] - blur[c]); // k=1
            out[idx + c] = v < 0 ? 0 : (v > 255 ? 255 : v);
        }
    }

    __global__ void erodeDilateKernel(const unsigned char* in,
        unsigned char* out,
        int w, int h, int ch,
        bool doErode)        // true=erode, false=dilate
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= w || y >= h) return;

        int idx = (y * w + x) * ch;

        for (int c = 0;c < ch;++c) {
            int extremum = in[idx + c];
            for (int ky = -1;ky <= 1;++ky)
                for (int kx = -1;kx <= 1;++kx) {
                    int nx = x + kx, ny = y + ky;
                    if (nx >= 0 && ny >= 0 && nx < w && ny < h) {
                        int v = in[(ny * w + nx) * ch + c];
                        extremum = doErode ? min(extremum, v) : max(extremum, v);
                    }
                }
            out[idx + c] = static_cast<unsigned char>(extremum);
        }
    }

    static void allocAndCopy(const Image& A,
        unsigned char*& d_A,
        unsigned char*& d_B,
        unsigned char*& d_O,
        const Image* B_ptr = nullptr)
    {
        size_t bytes = size_t(A.width) * A.height * A.channels;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_O, bytes);
        cudaMemcpy(d_A, A.data.data(), bytes, cudaMemcpyHostToDevice);

        if (B_ptr) {
            cudaMalloc(&d_B, bytes);
            cudaMemcpy(d_B, B_ptr->data.data(), bytes, cudaMemcpyHostToDevice);
        }
        else {
            d_B = nullptr;
        }
    }

    static void freeAll(unsigned char* d_A,
        unsigned char* d_B,
        unsigned char* d_O)
    {
        cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        cudaFree(d_O);
    }

    float launchAdd(const Image& A, const Image& B, Image& O)
    {
        unsigned char* d_A, * d_B, * d_O;
        allocAndCopy(A, d_A, d_B, d_O, &B);

        dim3 block(16, 16),
            grid((A.width + block.x - 1) / block.x,
                (A.height + block.y - 1) / block.y);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        addKernel << <grid, block >> > (d_A, d_B, d_O,
            A.width, A.height, A.channels);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        size_t bytes = size_t(A.width) * A.height * A.channels;
        cudaMemcpy(O.data.data(), d_O, bytes, cudaMemcpyDeviceToHost);

        freeAll(d_A, d_B, d_O);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return ms;
    }

    float launchSub(const Image& A, const Image& B, Image& O)
    {
        unsigned char* d_A, * d_B, * d_O;
        allocAndCopy(A, d_A, d_B, d_O, &B);

        dim3 block(16, 16),
            grid((A.width + block.x - 1) / block.x,
                (A.height + block.y - 1) / block.y);

        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);

        subKernel << <grid, block >> > (d_A, d_B, d_O,
            A.width, A.height, A.channels);

        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float ms = 0;
        cudaEventElapsedTime(&ms, s, e);

        size_t bytes = size_t(A.width) * A.height * A.channels;
        cudaMemcpy(O.data.data(), d_O, bytes, cudaMemcpyDeviceToHost);

        freeAll(d_A, d_B, d_O);
        cudaEventDestroy(s); cudaEventDestroy(e);
        return ms;
    }

    float launchSmooth(const Image& A, Image& O)
    {
        unsigned char* d_A, * d_B, * d_O;
        allocAndCopy(A, d_A, d_B, d_O);

        dim3 block(16, 16),
            grid((A.width + block.x - 1) / block.x,
                (A.height + block.y - 1) / block.y);

        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);

        smoothKernel << <grid, block >> > (d_A, d_O,
            A.width, A.height, A.channels);

        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float ms = 0; cudaEventElapsedTime(&ms, s, e);

        size_t bytes = size_t(A.width) * A.height * A.channels;
        cudaMemcpy(O.data.data(), d_O, bytes, cudaMemcpyDeviceToHost);

        freeAll(d_A, d_B, d_O);
        cudaEventDestroy(s); cudaEventDestroy(e);
        return ms;
    }

    float launchEnhance(const Image& A, Image& O)
    {
        unsigned char* d_A, * d_B, * d_O;
        allocAndCopy(A, d_A, d_B, d_O);

        dim3 block(16, 16),
            grid((A.width + block.x - 1) / block.x,
                (A.height + block.y - 1) / block.y);

        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);

        enhanceKernel << <grid, block >> > (d_A, d_O,
            A.width, A.height, A.channels);

        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float ms = 0; cudaEventElapsedTime(&ms, s, e);

        size_t bytes = size_t(A.width) * A.height * A.channels;
        cudaMemcpy(O.data.data(), d_O, bytes, cudaMemcpyDeviceToHost);

        freeAll(d_A, d_B, d_O);
        cudaEventDestroy(s); cudaEventDestroy(e);
        return ms;
    }

    float launchErode(const Image& A, Image& O)
    {
        unsigned char* d_A, * d_B, * d_O;
        allocAndCopy(A, d_A, d_B, d_O);

        dim3 block(16, 16),
            grid((A.width + block.x - 1) / block.x,
                (A.height + block.y - 1) / block.y);

        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);

        erodeDilateKernel << <grid, block >> > (d_A, d_O,
            A.width, A.height, A.channels,
            true);

        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float ms = 0; cudaEventElapsedTime(&ms, s, e);

        size_t bytes = size_t(A.width) * A.height * A.channels;
        cudaMemcpy(O.data.data(), d_O, bytes, cudaMemcpyDeviceToHost);

        freeAll(d_A, d_B, d_O);
        cudaEventDestroy(s); cudaEventDestroy(e);
        return ms;
    }

    float launchDilate(const Image& A, Image& O)
    {
        unsigned char* d_A, * d_B, * d_O;
        allocAndCopy(A, d_A, d_B, d_O);

        dim3 block(16, 16),
            grid((A.width + block.x - 1) / block.x,
                (A.height + block.y - 1) / block.y);

        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);

        erodeDilateKernel << <grid, block >> > (d_A, d_O,
            A.width, A.height, A.channels,
            false);

        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float ms = 0; cudaEventElapsedTime(&ms, s, e);

        size_t bytes = size_t(A.width) * A.height * A.channels;
        cudaMemcpy(O.data.data(), d_O, bytes, cudaMemcpyDeviceToHost);

        freeAll(d_A, d_B, d_O);
        cudaEventDestroy(s); cudaEventDestroy(e);
        return ms;
    }
}