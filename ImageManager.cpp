#include "ImageManager.hpp"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <tiffio.h> 
#include <cmath> 


#pragma pack(push,1)
struct BMPFileHeader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BMPInfoHeader {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

static std::string getExtension(const std::string& path) {
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return "";
    std::string ext = path.substr(pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

bool ImageManager::loadImage(const std::string& p, Image& img)
{
    auto ext = getExtension(p);
    if (ext == "bmp")  return loadBMP(p, img);
    if (ext == "tif" || ext == "tiff") return loadTIFF(p, img);
    std::cerr << "Unsupported format: " << ext << '\n';
    return false;
}

bool ImageManager::saveImage(const std::string& p, const Image& img)
{
    auto ext = getExtension(p);
    if (ext == "bmp")  return saveBMP(p, img);
    if (ext == "tif" || ext == "tiff") return saveTIFF(p, img);
    std::cerr << "Unsupported format: " << ext << '\n';
    return false;
}

bool ImageManager::loadBMP(const std::string& path, Image& img) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    BMPFileHeader fileHeader;
    BMPInfoHeader infoHeader;
    in.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    in.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

    if (fileHeader.bfType != 0x4D42) return false; // 'BM'
    if (infoHeader.biBitCount != 24) {
        std::cerr << "Only 24-bit BMP supported" << std::endl;
        return false;
    }

    img.width = infoHeader.biWidth;
    img.height = std::abs(infoHeader.biHeight);
    img.channels = 3;
    img.data.resize(img.width * img.height * img.channels);

    int rowSize = ((infoHeader.biBitCount * img.width + 31) / 32) * 4;
    int padding = rowSize - (img.width * 3);

    // BMP stores pixels bottom-up
    for (int y = 0; y < img.height; ++y) {
        int row = (infoHeader.biHeight > 0) ? (img.height - 1 - y) : y;
        for (int x = 0; x < img.width; ++x) {
            unsigned char bgr[3];
            in.read(reinterpret_cast<char*>(bgr), 3);
            int idx = (row * img.width + x) * img.channels;
            img.data[idx + 0] = bgr[2]; // R
            img.data[idx + 1] = bgr[1]; // G
            img.data[idx + 2] = bgr[0]; // B
        }
        in.seekg(padding, std::ios::cur);
    }
    in.close();
    return true;
}

bool ImageManager::saveBMP(const std::string& path, const Image& img) {
    if (img.channels != 3) {
        std::cerr << "Only 3-channel RGB supported for BMP output" << std::endl;
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    BMPFileHeader fileHeader;
    BMPInfoHeader infoHeader;
    int rowSize = ((24 * img.width + 31) / 32) * 4;
    int imgSize = rowSize * img.height;

    fileHeader.bfType = 0x4D42;
    fileHeader.bfSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + imgSize;
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;
    fileHeader.bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

    infoHeader.biSize = sizeof(BMPInfoHeader);
    infoHeader.biWidth = img.width;
    infoHeader.biHeight = img.height;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 24;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = imgSize;
    infoHeader.biXPelsPerMeter = 0;
    infoHeader.biYPelsPerMeter = 0;
    infoHeader.biClrUsed = 0;
    infoHeader.biClrImportant = 0;

    out.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    out.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));

    int padding = rowSize - (img.width * 3);
    unsigned char pad[3] = { 0, 0, 0 };

    // Write pixels bottom-up
    for (int y = 0; y < img.height; ++y) {
        int row = img.height - 1 - y;
        for (int x = 0; x < img.width; ++x) {
            int idx = (row * img.width + x) * img.channels;
            unsigned char bgr[3] = {
                img.data[idx + 2], // B
                img.data[idx + 1], // G
                img.data[idx + 0]  // R
            };
            out.write(reinterpret_cast<const char*>(bgr), 3);
        }
        out.write(reinterpret_cast<const char*>(pad), padding);
    }
    out.close();
    return true;
}

bool ImageManager::loadTIFF(const std::string& path, Image& img)
{
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    if (!tif) { std::cerr << "TIFFOpen failed\n"; return false; }

    uint32 w, h;
    uint16 spp, bpp, photo, planar;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bpp);
    TIFFGetFieldDefaulted(tif, TIFFTAG_PHOTOMETRIC, &photo);
    TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &planar);

    if (bpp != 8 || (photo != PHOTOMETRIC_RGB && photo != PHOTOMETRIC_MINISBLACK))
    {
        std::cerr << "Only 8-bit RGB or grayscale TIFF supported\n";
        TIFFClose(tif); return false;
    }
    img.width = w;
    img.height = h;
    img.channels = spp;              // 1 or 3
    img.data.resize(w * h * spp);

    // read line by line into img.data (interleaved)
    tsize_t scanlineSize = TIFFScanlineSize(tif);
    std::vector<uint8_t> line(scanlineSize);

    for (uint32 y = 0; y < h; ++y) {
        uint32 row = planar == PLANARCONFIG_CONTIG ? y : h - 1 - y;  // handle orientation if needed
        if (TIFFReadScanline(tif, line.data(), row, 0) < 0) {
            std::cerr << "TIFFReadScanline failed\n"; TIFFClose(tif); return false;
        }
        std::memcpy(&img.data[y * w * spp], line.data(), w * spp);
    }
    TIFFClose(tif);
    return true;
}

bool ImageManager::saveTIFF(const std::string& path, const Image& img)
{
    TIFF* tif = TIFFOpen(path.c_str(), "w");
    if (!tif) { std::cerr << "TIFFOpen write failed\n"; return false; }

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, img.width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, img.height);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, img.channels);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,
        img.channels == 1 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB);

    tsize_t linebytes = img.width * img.channels;
    std::vector<uint8_t> buf(linebytes);

    for (int y = 0; y < img.height; ++y)
    {
        std::memcpy(buf.data(), &img.data[y * linebytes], linebytes);
        if (TIFFWriteScanline(tif, buf.data(), y, 0) < 0) {
            std::cerr << "TIFFWriteScanline failed\n"; TIFFClose(tif); return false;
        }
    }
    TIFFClose(tif);
    return true;
}