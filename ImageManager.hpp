#pragma once
#include <string>
#include <vector>

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<unsigned char> data;
};

class ImageManager {
public:
    static bool loadImage(const std::string& path, Image& img);
    static bool saveImage(const std::string& path, const Image& img);

private:
    static bool loadBMP(const std::string& path, Image& img);
    static bool saveBMP(const std::string& path, const Image& img);

    static bool loadTIFF(const std::string& path, Image& img);
    static bool saveTIFF(const std::string& path, const Image& img);

    static bool loadPNG(const std::string& path, Image& img);
    static bool savePNG(const std::string& path, const Image& img);
};
