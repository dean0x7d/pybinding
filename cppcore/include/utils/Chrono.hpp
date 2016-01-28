#pragma once
#include <string>
#include <chrono>

namespace tbm {

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

    double seconds() const {
        return 1e-9 * static_cast<double>(elapsed.count());
    }
    
    std::string str() const;
    Chrono& print(std::string msg = "");

    friend std::string str(const Chrono& c) { return c.str(); }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> tic_time;
    std::chrono::nanoseconds elapsed;
};

} // namespace tbm
