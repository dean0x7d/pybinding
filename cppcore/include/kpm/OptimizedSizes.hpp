#pragma once
#include <algorithm>
#include <vector>
#include <cassert>

namespace cpb { namespace kpm {

struct Indices;

/**
 Optimal matrix sizes needed for KPM moment calculation. See `OptimizedHamiltonian`.
 */
class OptimizedSizes {
    std::vector<int> data; ///< optimal matrix sizes for the first few KPM iterations
    int offset = 0; ///< needed to correctly compute off-diagonal elements (i != j)

public:
    explicit OptimizedSizes(int system_size) { data = {system_size}; }
    OptimizedSizes(std::vector<int> sizes, Indices const& idx);

    /// Return an index into `data`, indicating the optimal system size for
    /// the calculation of KPM moment number `n` out of total `num_moments`
    int index(int n, int num_moments) const {
        assert(n < num_moments);

        auto const max_index = std::min(
            static_cast<int>(data.size()) - 1,
            num_moments / 2
        );

        if (n < max_index) {
            return n; // size grows in the beginning
        } else { // constant in the middle and shrinking near the end as reverse `n`
            return std::min(max_index, num_moments - 1 - n + offset);
        }
    }

    /// Return the optimal system size for KPM moment number `n` out of total `num_moments`
    int optimal(int n, int num_moments) const {
        return data[index(n, num_moments)];
    }

    /// Would calculating this number of moments ever do a full matrix-vector multiplication?
    bool uses_full_system(int num_moments) const {
        return static_cast<int>(data.size()) < num_moments / 2;
    }

    int operator[](int i) const { return data[i]; }

    std::vector<int> const& get_data() const { return data; }
    int get_offset() const { return offset; }
};

}} // namespace cpb::kpm
