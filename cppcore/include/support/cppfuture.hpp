#pragma once
#include <memory>
#include <type_traits>

// helper functions for use until C++14 brings this into std
namespace cpb { namespace std14 {

template<class T, class... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template<class T>
auto cbegin(const T& t) -> decltype(t.cbegin()) {
    return t.cbegin();
}

template<class T>
auto cend(const T& t) -> decltype(t.cend()) {
    return t.cend();
}

template <bool condition, class T = void>
using enable_if_t = typename std::enable_if<condition, T>::type;

template<bool condition, class If, class Else>
using conditional_t = typename std::conditional<condition, If, Else>::type;

template<class T>
using add_const_t = typename std::add_const<T>::type;

template<class T>
using decay_t = typename std::decay<T>::type;

} // namespace std14

namespace std17 {

template<class T>
constexpr std14::add_const_t<T>& as_const(T& x) noexcept { return x; }
template <class T>
void as_const(const T&&) = delete;

}} // namespace cpb::std17
