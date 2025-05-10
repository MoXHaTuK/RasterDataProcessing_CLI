#pragma once
#include <optional>
#include "Task.hpp"

class CLIParser {
public:
    std::optional<Task> parse(int argc, char* argv[]) const;
private:
    void printUsage() const;
};
