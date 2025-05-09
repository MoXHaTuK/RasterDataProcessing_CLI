#include <iostream>
#include <string>
#include "ImageManager.hpp"

// Declarations for CUDA host wrappers
void launchAdd(const Image& imgA, const Image& imgB, Image& imgOut);
void launchSub(const Image& imgA, const Image& imgB, Image& imgOut);
void launchSmooth(const Image&, Image&);
void launchEnhance(const Image&, Image&);
void launchErode(const Image&, Image&);
void launchDilate(const Image&, Image&);

void printUsage()
{
    std::cout <<
        "GPU Image Processor (BMP, PNG, TIFF, GeoTIFF)\n"
        "Two-input ops:\n"
        "  add | sub      <input1> <input2> <output>\n"
        "Single-input ops:\n"
        "  smooth|enhance|erode|dilate  <input>  <output>\n"
        "Supported extensions: .bmp .png .tif/.tiff (GeoTIFF OK)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) { printUsage(); return 1; }

    std::string op = argv[1];

    bool isTwoInput = (op == "add" || op == "sub");
    int  expected = isTwoInput ? 5 : 4;       // exe + op + files…

    if (argc != expected) { printUsage(); return 1; }

    std::string inPath1 = argv[2];
    std::string inPath2 = isTwoInput ? argv[3] : "";   // only for add/sub
    std::string outPath = isTwoInput ? argv[4] : argv[3];

    Image imgA, imgB, imgOut;
    if (!ImageManager::loadImage(inPath1, imgA))
    {
        std::cerr << "Failed to load '" << inPath1 << "'\n";
        return 1;
    }
    if (isTwoInput) {
        if (!ImageManager::loadImage(inPath2, imgB))
        {
            std::cerr << "Failed to load '" << inPath2 << "'\n";
            return 1;
        }
        if (imgA.width != imgB.width || imgA.height != imgB.height || imgA.channels != imgB.channels) {
            std::cerr << "Input images must match dimensions/channels.\n";
            return 1;
        }
    }

    imgOut.width = imgA.width;
    imgOut.height = imgA.height;
    imgOut.channels = imgA.channels;
    imgOut.data.resize(static_cast<size_t>(imgOut.width) * imgOut.height * imgOut.channels);

    if (op == "add") launchAdd(imgA, imgB, imgOut);
    else if (op == "sub") launchSub(imgA, imgB, imgOut);
    else if (op == "smooth")  launchSmooth(imgA, imgOut);
    else if (op == "enhance") launchEnhance(imgA, imgOut);
    else if (op == "erode")   launchErode(imgA, imgOut);
    else if (op == "dilate")  launchDilate(imgA, imgOut);
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
