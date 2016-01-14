#pragma once

namespace tbm {

/**
 Holds the start and end index of an array slice in 1 dimension
 */
struct SliceIndex {
    /// Entire array dimension
    SliceIndex() : start(0), end(-1) {}
    /// Single index -> intentionally implicit so that `array[7]` still works
    SliceIndex(int index) : start(index), end(index + 1) {}
    /// Slice range: to be used like `array[{2, 5}]`
    SliceIndex(int start, int end) : start(start), end(end) {}

    int start;
    int end;
};

/**
 Multidimensional slice
 */
template<int N>
class SliceIndexND {
    SliceIndex data[N];

public:
    template<class... SliceIndices>
    SliceIndexND(SliceIndices const&... indices) : data{indices...} {}

    constexpr int size() const { return N; }
    SliceIndex& operator[](int i) { return data[i]; }
    SliceIndex const& operator[](int i) const { return data[i]; }
};

using SliceIndex3D = SliceIndexND<3>;

} // namespace tbm
