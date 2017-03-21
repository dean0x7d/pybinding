#pragma once
#include "detail/config.hpp"

#include <algorithm>

namespace cpb {

/**
 Slice a Vector into pieces of `slice_size`
 */
template<class Vector>
class Sliced {
    struct Iterator {
        using Self = Iterator;
        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = Self const;
        using reference = value_type&;
        using pointer = value_type*;

        using InnerIt = typename Vector::const_iterator;
        InnerIt it, last;
        idx_t step;

        Iterator(InnerIt first, InnerIt last, idx_t step) : it(first), last(last), step(step) {}
        Iterator(InnerIt last) : it(last) {}

        InnerIt begin() const { return it; }
        InnerIt end() const { return std::min(it + step, last); }

        reference operator*() { return *this; }
        pointer operator->() { return this; }
        Self& operator++() { it = end(); return *this; }

        friend bool operator==(Self const& l, Self const& r) { return l.it == r.it; }
        friend bool operator!=(Self const& l, Self const& r) { return !(l == r); }
    };

public:
    Sliced(Vector const& vec, idx_t slice_size) : vec(vec), slice_size(slice_size) {}

    Iterator begin() const { return {vec.begin(), vec.end(), slice_size}; }
    Iterator end() const { return {vec.end()}; }

private:
    Vector const& vec;
    idx_t slice_size;
};

/// Iterate over slices of a vector
template<class Vector>
Sliced<Vector> sliced(Vector const& vec, idx_t slice_size) { return {vec, slice_size}; }

} // namespace cpb
