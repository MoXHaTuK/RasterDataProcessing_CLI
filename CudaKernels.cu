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

    // 3?3 box blur inline
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
                    extremum = doErode ? min(extremum, v)
                        : max(extremum, v);
                }
            }
        out[idx + c] = static_cast<unsigned char>(extremum);
    }
}

static void launchUnary(const Image& in, Image& out,
    void(*kernel)(const unsigned char*, unsigned char*,
        int, int, int),
    dim3 block = dim3(16, 16))
{
    int size = in.width * in.height * in.channels;
    unsigned char* d_in, * d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, in.data.data(), size, cudaMemcpyHostToDevice);

    dim3 grid((in.width + block.x - 1) / block.x,
        (in.height + block.y - 1) / block.y);

    kernel << <grid, block >> > (d_in, d_out,
        in.width, in.height, in.channels);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data.data(), d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);
}

void launchSmooth(const Image& A, Image& O)
{
    launchUnary(A, O, smoothKernel);
}

void launchEnhance(const Image& A, Image& O)
{
    launchUnary(A, O, enhanceKernel);
}

void launchErode(const Image& A, Image& O)
{
    launchUnary(A, O, [](const unsigned char* in, unsigned char* out,
        int w, int h, int ch) {
            erodeDilateKernel << <1, 1 >> > (in, out, w, h, ch, true);
        });
}

void launchDilate(const Image& A, Image& O)
{
    launchUnary(A, O, [](const unsigned char* in, unsigned char* out,
        int w, int h, int ch) {
            erodeDilateKernel << <1, 1 >> > (in, out, w, h, ch, false);
        });
}