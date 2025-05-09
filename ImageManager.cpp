#include "ImageManager.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <png.h>
#include <tiffio.h>
//#include <geotiff.h>
//#include <geo_normalize.h>


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

static std::string toLowerExt(const std::string& path)
{
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return "";
    std::string ext = path.substr(pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return ext;
}

static std::string getExtension(const std::string& path) {
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return "";
    std::string ext = path.substr(pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

bool ImageManager::loadImage(const std::string& p, Image& img)
{
    std::string ext = toLowerExt(p);
    if (ext=="bmp")  return loadBMP (p,img);
    if (ext=="png")  return loadPNG (p,img);
    if (ext=="tif" || ext=="tiff") return loadTIFF(p,img);   // includes GeoTIFF
    std::cerr << "Unsupported format: " << ext << '\n';
    return false;
}
bool ImageManager::saveImage(const std::string& p,const Image& img)
{
    std::string ext = toLowerExt(p);
    if (ext=="bmp")  return saveBMP (p,img);
    if (ext=="png")  return savePNG (p,img);
    if (ext=="tif" || ext=="tiff") return saveTIFF(p,img);
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

    if (fileHeader.bfType != 0x4D42) return false;
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

bool ImageManager::loadPNG(const std::string& path, Image& img)
{
    FILE* fp = nullptr; 
    if (fopen_s(&fp, path.c_str(), "rb") || !fp) {
        std::cerr << "Cannot open PNG\n"; 
        return false;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { fclose(fp); return false; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, nullptr, nullptr); fclose(fp); return false; }

    if (setjmp(png_jmpbuf(png))) { png_destroy_read_struct(&png, &info, nullptr); fclose(fp); return false; }

    png_init_io(png, fp);
    png_read_info(png, info);

    png_uint_32 w, h; int bitDepth, colorType;
    png_get_IHDR(png, info, &w, &h, &bitDepth, &colorType, nullptr, nullptr, nullptr);
    if (bitDepth != 8) {
        std::cerr << "Only 8-bit PNG supported\n";
        png_destroy_read_struct(&png, &info, nullptr); fclose(fp); return false;
    }

    if (colorType == PNG_COLOR_TYPE_PALETTE)   png_set_palette_to_rgb(png);
    if (colorType == PNG_COLOR_TYPE_GRAY && bitDepth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (colorType == PNG_COLOR_TYPE_GRAY || colorType == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    img.width = static_cast<int>(w);
    img.height = static_cast<int>(h);
    img.channels = png_get_channels(png, info);    // will be 3 (RGB) or 4 (RGBA)
    if (img.channels == 4) { png_set_strip_alpha(png); img.channels = 3; }

    img.data.resize(w * h * img.channels);

    std::vector<png_bytep> rows(h);
    for (size_t y = 0;y < h;++y) rows[y] = img.data.data() + y * w * img.channels;
    png_read_image(png, rows.data());

    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);
    return true;
}

bool ImageManager::savePNG(const std::string& path, const Image& img)
{
    FILE* fp = nullptr;
    if (fopen_s(&fp, path.c_str(), "rb") || !fp) {
        std::cerr << "Cannot open PNG\n";
        return false;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { fclose(fp); return false; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, nullptr); fclose(fp); return false; }

    if (setjmp(png_jmpbuf(png))) { png_destroy_write_struct(&png, &info); fclose(fp); return false; }

    png_init_io(png, fp);
    png_set_IHDR(png, info, img.width, img.height,
        8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png, info);

    std::vector<png_bytep> rows(img.height);
    for (int y = 0;y < img.height;++y) rows[y] = const_cast<png_bytep>(
        img.data.data() + y * img.width * img.channels);
    png_write_image(png, rows.data());
    png_write_end(png, nullptr);

    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return true;
}

bool ImageManager::loadTIFF(const std::string& path, Image& img)
{
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    if (!tif) { std::cerr << "TIFFOpen failed\n"; return false; }

    uint32_t w, h; uint16_t spp, bpp, photo;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bpp);
    TIFFGetFieldDefaulted(tif, TIFFTAG_PHOTOMETRIC, &photo);
    if (bpp != 8) { std::cerr << "Only 8-bit TIFF supported\n"; TIFFClose(tif); return false; }

    img.width = w; img.height = h; img.channels = spp;
    img.data.resize(static_cast<size_t>(w) * h * spp);

    tsize_t linebytes = TIFFScanlineSize(tif);
    std::vector<uint8_t> buf(linebytes);
    for (uint32_t y = 0;y < h;++y) {
        if (TIFFReadScanline(tif, buf.data(), y, 0) < 0) { TIFFClose(tif); return false; }
        std::memcpy(&img.data[y * linebytes], buf.data(), linebytes);
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
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, img.channels == 1 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB);

    tsize_t stride = static_cast<tsize_t>(img.width) * img.channels;
    for (int y = 0;y < img.height;++y) {
        if (TIFFWriteScanline(tif,
            const_cast<uint8_t*>(&img.data[y * stride]), y, 0) < 0) {
            TIFFClose(tif); return false;
        }
    }
    TIFFClose(tif);
    return true;
}