#include "system/CompressedSublattices.hpp"

namespace cpb {

CompressedSublattices::CompressedSublattices(ArrayXi const& alias_ids, ArrayXi const& site_counts,
                                             ArrayXi const& orbital_counts)
    : data(alias_ids.size()) {
    for (auto i = size_t{0}; i < data.size(); ++i) {
        data[i].id = SiteID{alias_ids[i]};
        data[i].num_sites = site_counts[i];
        data[i].num_orbitals = orbital_counts[i];
    }
}

void CompressedSublattices::add(SiteID id, idx_t norb, idx_t count) {
    if (data.empty() || data.back().id != id) {
        data.push_back({id, static_cast<storage_idx_t>(count), static_cast<storage_idx_t>(norb)});
    } else {
        data.back().num_sites += static_cast<storage_idx_t>(count);
    }
}

void CompressedSublattices::filter(VectorX<bool> const& keep) {
    using std::begin;

    auto new_counts = std::vector<storage_idx_t>();
    new_counts.reserve(data.size());

    for (auto const& sub : *this) {
        new_counts.push_back(std::accumulate(begin(keep) + sub.sys_start(),
                                             begin(keep) + sub.sys_end(), storage_idx_t{0}));
    }

    for (auto i = size_t{0}; i < data.size(); ++i) {
        data[i].num_sites = new_counts[i];
    }
}

void CompressedSublattices::verify(idx_t num_sites) const {
    using std::begin; using std::end;

    auto const alias_ids_are_unique = [&]{
        // alias_ids: [1, 0, 2] --> OK
        //            [1, 0, 2, 1] --> Bad, repeating ID
        auto ids = alias_ids();
        std::sort(begin(ids), end(ids));
        auto const unique_size = std::unique(begin(ids), end(ids)) - begin(ids);
        return static_cast<size_t>(unique_size) == data.size();
    }();

    auto const is_sorted_by_orb_count = [&]{
        auto const norb = orbital_counts();
        return std::is_sorted(begin(norb), end(norb));
    }();

    if (decompressed_size() != num_sites || !alias_ids_are_unique || !is_sorted_by_orb_count) {
        throw std::runtime_error("CompressedSublatticeIDs: this should never happen");
    }
}

idx_t CompressedSublattices::start_index(idx_t num_orbitals) const {
    for (auto const& sub : *this) {
        if (sub.num_orbitals() == num_orbitals) {
            return sub.sys_start();
        }
    }
    throw std::runtime_error("CompressedSublattices::start_index(): invalid num_orbitals");
}

idx_t CompressedSublattices::decompressed_size() const {
    return std::accumulate(data.begin(), data.end(), idx_t{0}, [](idx_t n, Element const& v) {
        return n + v.num_sites;
    });
}

ArrayX<storage_idx_t> CompressedSublattices::decompressed() const {
    auto sublattices = ArrayX<storage_idx_t>(decompressed_size());
    for (auto const& sub : *this) {
        sublattices.segment(sub.sys_start(), sub.num_sites()).setConstant(sub.id().value());
    }
    return sublattices;
}

ArrayXi CompressedSublattices::alias_ids() const {
    auto result = ArrayXi(static_cast<idx_t>(data.size()));
    std::transform(data.begin(), data.end(), result.data(),
                   [](Element const& v) { return v.id.value(); });
    return result;
}

ArrayXi CompressedSublattices::site_counts() const {
    auto result = ArrayXi(static_cast<idx_t>(data.size()));
    std::transform(data.begin(), data.end(), result.data(),
                   [](Element const& v) { return v.num_sites; });
    return result;
}

ArrayXi CompressedSublattices::orbital_counts() const {
    auto result = ArrayXi(static_cast<idx_t>(data.size()));
    std::transform(data.begin(), data.end(), result.data(),
                   [](Element const& v) { return v.num_orbitals; });
    return result;
}

} // namespace cpb
