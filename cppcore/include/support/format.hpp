#pragma once
#include "support/cpp14.hpp"
#include <string>
#include <algorithm>
#include <chrono>
#include <cctype>

#if defined(_MSC_VER) && _MSC_VER <= 1800 // VS2013 does not have a C99 compliant snprintf
# include <cstdarg>
# define snprintf c99_snprintf

inline int c99_snprintf(char* str, size_t size, const char* format, ...) {
    int count = -1;
    va_list ap;

    va_start(ap, format);
    if (size != 0)
        count = _vsnprintf_s(str, size, _TRUNCATE, format, ap);
    if (count == -1)
        count = _vscprintf(format, ap);
    va_end(ap);

    return count;
}
#endif

namespace fmt {

inline void print(const std::string& str, bool flush = false) {
    printf("%s", str.c_str());
    if (flush)
        fflush(stdout);
}

inline void println(const std::string& str, bool flush = false) {
    print(str);
    printf("\n");
    if (flush)
        fflush(stdout);
}

// specialize for desired class
inline std::string str(const std::string& s) { return s; }

namespace detail {
    using tbm::cpp14::enable_if_t;

    inline const char* norm_arg(const std::string& x) { return x.c_str(); }

    template<class T> enable_if_t<std::is_integral<T>::value, std::int64_t> norm_arg(T x) { return x; }
    template<class T> enable_if_t<std::is_floating_point<T>::value, double> norm_arg(T x) { return x; }
    template<class T> enable_if_t<std::is_pointer<T>::value, T> norm_arg(T x) { return x; }

    template<class T> enable_if_t<
        !std::is_integral<T>::value &&
        !std::is_floating_point<T>::value &&
        !std::is_pointer<T>::value,
        const T*
    >
    inline norm_arg(const T& x) { return &x; }

    template<class R, class P>
    inline auto norm_arg(const std::chrono::duration<R, P>& x) -> decltype(norm_arg(x.count())) {
        return norm_arg(x.count());
    }

    template<class T>
    inline std::string conversion_specifier(const T* t) { return ".0s" + str(*t); }
    inline constexpr char conversion_specifier(std::int64_t) { return 'd'; }
    inline constexpr char conversion_specifier(double) { return 'f'; }
    inline constexpr char conversion_specifier(const char*) { return 's'; }

    template<class T> inline bool check_specifier(char) { return false; }
    template<> inline bool check_specifier<std::int64_t>(char type) { return type == 'd' || type == 'i'; }
    template<> inline bool check_specifier<double>(char type) { return type == 'f' || type == 'g' || type == 'e'; }
    template<> inline bool check_specifier<const char*>(char type) { return type == 's'; }

    struct convert_and_check_format {
        template <class... Ts>
        convert_and_check_format(const std::string& fmt, const Ts&... ts)
            : _begin{fmt.begin()}, _end{fmt.end()}
        {
            format.reserve(fmt.size());
            convert(ts...);
        }

        const char* c_str() const { return format.c_str(); }

    private:
        using iter_t = std::string::const_iterator;

        iter_t append(iter_t from, const iter_t to) {
            for (; from != to; ++from) {
                if (*from == '}' && (from + 1) != to && *(from + 1) == '}')
                    continue; // only print one close brace "}}"

                format.push_back(*from);
                if (*from == '%') // escape % by doubling it
                    format.push_back('%');
            }
            return to;
        }

        void convert() {
            // no arguments left, we don't want to find any more format specifiers
            it = std::find(_begin, _end, '{');
            if (it != _end && it + 1 != _end) {
                ++it;
                if (*it == '{') { // skip double brace "{{"
                    append(_begin, it);
                    _begin = it + 1;
                }
                else {
                    throw std::logic_error{"Too many format specifiers."};
                }
            }

            append(_begin, _end); // copy everything that's left
        }

        template <class T, class... Ts>
        void convert(const T& t, const Ts&... ts) {
            // look for a *single* open brace
            do {
                it = std::find(_begin, _end, '{');
                if (it == _end || it + 1 == _end)
                    throw std::logic_error{"Too few format specifiers."};

                ++it;
                if (*it == '{') { // skip double brace "{{"
                    append(_begin, it);
                    _begin = it + 1;
                }
            } while (*it == '{');

            // copy everything before the open brace
            _begin = append(_begin, it - 1);

            // replace '{' with '%'
            format.push_back('%');

            // look for close brace
            it = std::find(_begin, _end, '}');
            if (it == _end)
                throw std::logic_error{"Unclosed brace fromat specifier."};

            _begin = std::find(_begin, it, ':');
            if (_begin == it || _begin + 1 == it) { // automatically deduce type
                format += detail::conversion_specifier(t);
            }
            else { // check type
                if (!detail::check_specifier<T>(*(it - 1)))
                    throw std::logic_error{"Invalid format specifier: " + std::string(it - 1, it)};

                format.append(_begin + 1, it);
            }

            _begin = it + 1;
            convert(ts...); // this is not recursion (calling different function template)
        }

    private:
        iter_t it, _begin;
        const iter_t _end;
        std::string format;
    };
}

template<class... Ts>
std::string format(std::string fmt, Ts&&... ts) {
    const auto new_fmt = detail::convert_and_check_format(fmt, detail::norm_arg(ts)...);

    auto size = 2 * fmt.size();
    do {
        fmt.resize(size + 1);
        auto ret = snprintf(&fmt[0], fmt.size(), new_fmt.c_str(), detail::norm_arg(ts)...);
        if (ret >= 0)
            size = ret;
        else
            throw std::runtime_error{"Error while using snprintf() in fmt::format()."};
    } while (size > fmt.size());

    fmt.resize(size);
    return fmt;
}

/**
 Convert number to string with SI suffix, e.g.: 14226 -> 14.2k, 5395984 -> 5.39M
 */
inline std::string with_suffix(double number) {
    struct Pair {
        double value;
        char const* suffix;
    };
    static constexpr Pair mapping[] = {{1e9, "G"}, {1e6, "M"}, {1e3, "k"}};

    auto const result = [&]{
        for (auto const& bucket : mapping) {
            if (number > bucket.value)
                return Pair{number / bucket.value, bucket.suffix};
        }
        return Pair{number, ""};
    }();

    return format("{value:.3g}{suffix}", result.value, result.suffix);
}

} // namespace fmt
