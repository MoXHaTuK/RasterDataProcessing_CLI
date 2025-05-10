#pragma once
#include <string>
#include <chrono>

struct Task {
    std::string operation;
    std::string input1;
    std::string input2;
    std::string output;

    std::chrono::high_resolution_clock::time_point startTS, endTS;
    float gpuMs = 0.0f;
};
