#include <iostream>
#include <string>
#include "ImageManager.hpp"

// Declarations for CUDA host wrappers
void launchAdd(const Image& imgA, const Image& imgB, Image& imgOut);
void launchSub(const Image& imgA, const Image& imgB, Image& imgOut);

void printUsage() {
    std::cout << "Usage: CudaBmpProcessor <add|sub> <input1.bmp> <input2.bmp> <output.bmp>\n";
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printUsage();
        return 1;
    }

    std::string op = argv[1];
    std::string inPath1 = argv[2];
    std::string inPath2 = argv[3];
    std::string outPath = argv[4];

    Image imgA, imgB, imgOut;
    if (!ImageManager::loadImage(inPath1, imgA) || !ImageManager::loadImage(inPath2, imgB)) {
        std::cerr << "Failed to load input images." << std::endl;
        return 1;
    }
    if (imgA.width != imgB.width || imgA.height != imgB.height || imgA.channels != imgB.channels) {
        std::cerr << "Input images must have same dimensions and channels." << std::endl;
        return 1;
    }

    imgOut.width = imgA.width;
    imgOut.height = imgA.height;
    imgOut.channels = imgA.channels;
    imgOut.data.resize(imgOut.width * imgOut.height * imgOut.channels);

    if (op == "add") {
        launchAdd(imgA, imgB, imgOut);
    }
    else if (op == "sub") {
        launchSub(imgA, imgB, imgOut);
    }
    else {
        printUsage();
        return 1;
    }

    if (!ImageManager::saveImage(outPath, imgOut)) {
        std::cerr << "Failed to save output image." << std::endl;
        return 1;
    }

    std::cout << "Operation " << op << " completed successfully. Output: " << outPath << std::endl;
    return 0;
}
