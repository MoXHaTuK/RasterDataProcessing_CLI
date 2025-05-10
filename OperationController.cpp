#include "OperationController.hpp"
#include "ImageManager.hpp"
#include "KernelWrappers.hpp"
#include <iostream>
#include <chrono>

using clk = std::chrono::high_resolution_clock;

static void prepareOut(const Image& in, Image& out)
{
    out.width = in.width; out.height = in.height; out.channels = in.channels;
    out.data.resize(static_cast<size_t>(in.width) * in.height * in.channels);
}

void OperationController::runTask(Task& t) const
{
    Image A, B, O;

    if (!ImageManager::loadImage(t.input1, A)) {
        std::cerr << "Cannot load " << t.input1 << "\n"; return;
    }
    bool twoIn = !t.input2.empty();
    if (twoIn) {
        if (!ImageManager::loadImage(t.input2, B)) { std::cerr << "Cannot load " << t.input2 << "\n"; return; }
        if (A.width != B.width || A.height != B.height || A.channels != B.channels) {
            std::cerr << "Dimension mismatch\n"; return;
        }
    }
    prepareOut(A, O);

    t.startTS = clk::now();

    if (t.operation == "add")     t.gpuMs = GPU::launchAdd(A, B, O);
    else if (t.operation == "sub")     t.gpuMs = GPU::launchSub(A, B, O);
    else if (t.operation == "smooth")  t.gpuMs = GPU::launchSmooth(A, O);
    else if (t.operation == "enhance") t.gpuMs = GPU::launchEnhance(A, O);
    else if (t.operation == "erode")   t.gpuMs = GPU::launchErode(A, O);
    else if (t.operation == "dilate")  t.gpuMs = GPU::launchDilate(A, O);
    else { std::cerr << "Unknown op\n"; return; }

    t.endTS = clk::now();

    if (!ImageManager::saveImage(t.output, O)) {
        std::cerr << "Save failed\n"; return;
    }
    auto cpuMs = std::chrono::duration<double, std::milli>(t.endTS - t.startTS).count();
    std::cout << "Done '" << t.operation << "'  CPU " << cpuMs << " ms  GPU " << t.gpuMs << " ms\n";
}
