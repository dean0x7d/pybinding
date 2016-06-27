#pragma once
#include <algorithm>
#include <numeric>

namespace cpb {

/**
 Holds the start and end index of an array slice in 1 dimension
 */
struct SliceIndex {
    int start;
    int end;

    /// Entire array dimension
    SliceIndex() : start(0), end(-1) {}
    /// Single index -> intentionally implicit so that `array[7]` still works
    SliceIndex(int index) : start(index), end(index + 1) {}
    /// Slice range: to be used like `array[{2, 5}]`
    SliceIndex(int start, int end) : start(start), end(end) {}

    int size() const { return (end > start) ? (end - start) : 0; }

    SliceIndex& operator+=(int n) { start += n; end += n; return *this; }
    SliceIndex& operator-=(int n) { start -= n; end -= n; return *this; }
    SliceIndex& operator++() { operator+=(1); return *this; }
    SliceIndex operator++(int) { auto const copy = *this; operator++(); return copy; }
    SliceIndex& operator--() { operator-=(1); return *this; }
    SliceIndex operator--(int) { auto const copy = *this; operator--(); return copy; }

    friend bool operator==(SliceIndex const& l, SliceIndex const& r) {
        return (l.start == r.start) && (l.end == r.end);
    }
    friend bool operator!=(SliceIndex const& l, SliceIndex const& r) { return !(l == r); }
};

/**
 Multidimensional slice
 */
template<int N>
class SliceIndexND {
    SliceIndex data[N];

public:
    SliceIndexND() = default;
    SliceIndexND(std::initializer_list<SliceIndex> indices) {
        std::copy_n(indices.begin(), N, data);
    }

    int size() const {
        return std::accumulate(std::begin(data), std::end(data), 1,
                               [](int a, SliceIndex b) { return a * b.size(); });
    }

    constexpr int ndims() const { return N; }
    SliceIndex& operator[](int i) { return data[i]; }
    SliceIndex const& operator[](int i) const { return data[i]; }
};

using SliceIndex3D = SliceIndexND<3>;

} // namespace cpb
