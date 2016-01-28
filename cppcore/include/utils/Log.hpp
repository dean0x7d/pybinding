#pragma once
#include <string>

namespace tbm {

class Log {
public:
    /// info
    static void i(const std::string& str, bool new_line = true, int width = 0);
    /// debug
#ifdef DEBUG
    static void d(const std::string& str, bool new_line = true, int width = 0) {
        i(str, new_line, width);
    }
#else
    static void d(const std::string&, bool = true, int = 0) { /* pass */ }
#endif
};

} // namespace tbm
