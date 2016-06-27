#include "utils/Chrono.hpp"
#include "support/format.hpp"
using namespace std::chrono;
using namespace cpb;

std::string Chrono::str() const {
    auto ret = std::string{};

    if (elapsed < milliseconds{1}) {
        ret = fmt::format("{:.2f}ms", duration_cast<duration<float, std::milli>>(elapsed).count());
    } else if (elapsed < milliseconds{10}) {
        ret = fmt::format("{:.1f}ms", duration_cast<duration<float, std::milli>>(elapsed).count());
    } else if (elapsed < milliseconds{100}) {
        ret = fmt::format("{}ms", duration_cast<milliseconds>(elapsed).count());
    } else if (elapsed < seconds{10}) {
        ret = fmt::format("{:.2f}s", duration_cast<duration<float>>(elapsed).count());
    } else if (elapsed < seconds{20}) {
        ret = fmt::format("{:.1f}s", duration_cast<duration<float>>(elapsed).count());
    } else if (elapsed < seconds{60}) {
        ret = fmt::format("{}s", duration_cast<seconds>(elapsed).count());
    } else { // elapsed >= minutes{1}
        auto const min = duration_cast<minutes>(elapsed);
        auto const sec = duration_cast<seconds>(elapsed) - min;

        if (min < minutes{60}) {
            ret = fmt::format("{}:{:02}", min.count(), sec.count());
        } else { // elapsed >= hours{1}
            auto const hr = duration_cast<hours>(min);
            ret = fmt::format("{}:{:02}:{:02}", hr.count(), (min - hr).count(), sec.count());
        }
    }

    return ret;
}

Chrono& Chrono::print(std::string msg) {
    if (!msg.empty())
        msg += ": ";

    fmt::print("{}{}\n", msg, str());
    return *this;
}
