#pragma once
#include "detail/config.hpp"
#include "support/cppfuture.hpp"

namespace cpb { namespace detail {

/**
 Type-safe integer alias
 */
template<class Tag, class T = storage_idx_t>
class OpaqueIntegerAlias {
    using Self = OpaqueIntegerAlias;

    // Don't create duplicate constructors
    static constexpr auto has_constructor = std::is_same<T, idx_t>::value
                                            || std::is_same<T, size_t>::value;

public:
    OpaqueIntegerAlias() = default;
    explicit OpaqueIntegerAlias(idx_t value) : _value(static_cast<T>(value)) {}
    explicit OpaqueIntegerAlias(size_t value) : _value(static_cast<T>(value)) {}

    template<class U, class = std14::enable_if_t<std::is_same<U, T>::value && !has_constructor>>
    explicit OpaqueIntegerAlias(U value) : _value(value) {}

    template<class OtherTag>
    explicit OpaqueIntegerAlias(OpaqueIntegerAlias<OtherTag, T> const& other)
        : _value(other.value()) {}

    T value() const { return _value; }
    template<class U> U as() const { return static_cast<U>(_value); }

    friend bool operator==(Self a, Self b) { return a._value == b._value; }
    friend bool operator!=(Self a, Self b) { return !(a == b); }
    friend bool operator< (Self a, Self b) { return a._value < b._value; }
    friend bool operator> (Self a, Self b) { return b < a; }
    friend bool operator>=(Self a, Self b) { return !(a < b); }
    friend bool operator<=(Self a, Self b) { return !(a > b); }

private:
    T _value;
};

}} // namespace cpb::detail
