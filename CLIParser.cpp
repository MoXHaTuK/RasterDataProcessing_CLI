#include "CLIParser.hpp"
#include <iostream>

void CLIParser::printUsage() const
{
    std::cout <<
        "GPU Image Processor (BMP, PNG, TIFF, GeoTIFF)\n"
        "Two-input:  add|sub   <in1> <in2> <out>\n"
        "Single-in : smooth|enhance|erode|dilate <in> <out>\n";
}

std::optional<Task> CLIParser::parse(int argc, char* argv[]) const
{
    if (argc < 4) { printUsage(); return std::nullopt; }

    Task t;
    t.operation = argv[1];

    bool twoIn = (t.operation == "add" || t.operation == "sub");
    int  need = twoIn ? 5 : 4;
    if (argc != need) { printUsage(); return std::nullopt; }

    t.input1 = argv[2];
    if (twoIn) { t.input2 = argv[3]; t.output = argv[4]; }
    else { t.output = argv[3]; }

    return t;
}
