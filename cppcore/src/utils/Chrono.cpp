#include "utils/Chrono.hpp"
#include "support/format.hpp"
using namespace std::chrono;
using namespace tbm;

std::string Chrono::str() const {
    using fmt::format;
    using std::chrono::seconds;

    auto ret = std::string{};

    if (elapsed < milliseconds{1}) {
        ret = format("{:.2f}ms", duration_cast<duration<float, std::milli>>(elapsed));
    }
    else if (elapsed < milliseconds{10}) {
        ret = format("{:.1f}ms", duration_cast<duration<float, std::milli>>(elapsed));
    }
    else if (elapsed < milliseconds{100}) {
        ret = format("{}ms", duration_cast<milliseconds>(elapsed));
    }
    else if (elapsed < seconds{10}) {
        ret = format("{:.2f}s", duration_cast<duration<float>>(elapsed));
    }
    else if (elapsed < seconds{20}) {
        ret = format("{:.1f}s", duration_cast<duration<float>>(elapsed));
    }
    else if (elapsed < seconds{60}) {
        ret = format("{}s", duration_cast<seconds>(elapsed));
    }
    else { // elapsed >= minutes{1}
        auto min = duration_cast<minutes>(elapsed);
        auto sec = duration_cast<seconds>(elapsed) - min;

        if (min < minutes{60}) {
            ret = format("{}:{:02i}", min, sec);
        }
        else { // elapsed >= hours{1}
            auto hr = duration_cast<hours>(min);
            min -= hr;
            ret = format("{}:{:02i}:{:02i}", hr, min, sec);
        }
    }

    return ret;
}

Chrono& Chrono::print(std::string msg) {
    if (!msg.empty())
        msg += ": ";

    fmt::println(fmt::format("{}{}", msg, str()));
    return *this;
}
