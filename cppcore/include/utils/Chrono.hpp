#pragma once
#include <string>
#include <chrono>
#include <ostream>

namespace cpb {

/**
 High resolution timer (below 1 microsecond accuracy).
 */
class Chrono {
public:
    Chrono() { tic(); };

    void tic() {
        tic_time = std::chrono::high_resolution_clock::now();
    }

    Chrono& toc() {
        elapsed = std::chrono::high_resolution_clock::now() - tic_time;
        return *this;
    }

    template<class Fn>
    Chrono& timeit(Fn lambda) {
        tic(); lambda(); toc();
        return *this;
    }

    double elapsed_seconds() const {
        return 1e-9 * static_cast<double>(elapsed.count());
    }
    
    std::string str() const;
    Chrono& print(std::string msg = "");

    friend std::ostream& operator<<(std::ostream& os, Chrono const& chrono) {
        os << chrono.str();
        return os;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> tic_time;
    std::chrono::nanoseconds elapsed{0};
};

} // namespace cpb
