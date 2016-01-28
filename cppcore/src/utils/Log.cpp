#include "utils/Log.hpp"
#include <iostream>
#include <iomanip>
using namespace tbm;

void Log::i(const std::string& str, bool new_line, int width)
{
    if (width) {
        std::cout.setf(std::ios::left);
        std::cout.width(width);
    }
    std::cout << str;

    if (new_line)
        std::cout << std::endl;
    std::cout.width(0);
}
