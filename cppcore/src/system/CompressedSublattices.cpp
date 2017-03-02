#include "system/CompressedSublattices.hpp"

namespace cpb {

void CompressedSublattices::add(sub_id id) {
    if (alias_ids.empty() || alias_ids.back() != id) {
        alias_ids.push_back(id);
        site_counts.push_back(1);
    } else {
        site_counts.back() += 1;
    }
}

void CompressedSublattices::verify(idx_t num_sites) const {
    // alias_ids: [1, 0, 2] --> OK
    //            [1, 0, 2, 1] --> Bad, repeating ID
    auto unique_ids = alias_ids;
    std::sort(unique_ids.begin(), unique_ids.end());
    unique_ids.erase(std::unique(unique_ids.begin(), unique_ids.end()),
                     unique_ids.end());

    auto const actual_nsites = std::accumulate(site_counts.begin(), site_counts.end(), idx_t{0});
    if (unique_ids.size() != alias_ids.size() || actual_nsites != num_sites) {
        throw std::runtime_error("CompressedSublatticeIDs: this should never happen");
    }
}

ArrayX<sub_id> CompressedSublattices::decompress() const {
    auto const total_size = std::accumulate(site_counts.begin(), site_counts.end(), idx_t{0});
    auto sublattices = ArrayX<sub_id>(total_size);
    auto start = idx_t{0};
    for (auto const& sub : *this) {
        sublattices.segment(start, sub.site_count()).setConstant(sub.alias_id());
        start += sub.site_count();
    }
    return sublattices;
}

} // namespace cpb
