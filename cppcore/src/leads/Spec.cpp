#include "leads/Spec.hpp"
#include "system/Foundation.hpp"

namespace cpb { namespace leads {

Spec::Spec(int direction, Shape const& shape)
    : axis(abs(direction) - 1),
      sign(direction != 0 ? direction / abs(direction) : 0),
      shape(shape) {}

void create_attachment_area(Foundation& foundation, Spec const& spec) {
    auto const size = foundation.get_size();
    auto const step = -spec.sign;
    auto const end = (step > 0) ? size[spec.axis] : -1;

    auto junction = detail::Junction(foundation, spec);
    auto slice = foundation[junction.slice_index];

    // Fill in the foundation until all lead sites can be connected to foundation sites
    for (; slice[spec.axis] != end; slice[spec.axis] += step) {
        for (auto& site : slice) {
            if (!junction.is_valid[site.get_slice_idx()])
                continue;

            if (!site.is_valid()) {
                // Empty sites should be filled in for the lead
                site.set_valid(true);
            } else {
                // Stop once we hit an existing site on the lead's path
                junction.is_valid[site.get_slice_idx()] = false;
            }
        }

        if (none_of(junction.is_valid)) {
            break;
        }
    }

    if (slice[spec.axis] == end) {
        throw std::runtime_error("Can't attach lead: partially misses main structure");
    }
}

namespace detail {

SliceIndex3D shape_slice(Foundation const& foundation, Shape const& shape) {
    auto const size = foundation.get_size();
    auto const foundation_bounds = foundation.get_bounds();
    auto const lead_bounds = cpb::detail::find_bounds(shape, foundation.get_lattice());

    auto slice_index = SliceIndex3D();
    for (auto i = 0; i < slice_index.ndims(); ++i) {
        auto const lead_start = lead_bounds.first[i] - foundation_bounds.first[i];
        auto const lead_end = (lead_bounds.second[i] + 1) - foundation_bounds.first[i];
        slice_index[i] = {std::max(lead_start, 0), std::min(lead_end, size[i])};
    }
    return slice_index;
}

SliceIndex3D attachment_slice(Foundation const& foundation, Spec const& spec) {
    auto const size = foundation.get_size();
    auto const step = -spec.sign;
    auto const start = (step > 0) ? 0 : size[spec.axis] - 1;
    auto const end = (step > 0) ? size[spec.axis] : -1;

    auto slice_index = shape_slice(foundation, spec.shape);
    auto slice = foundation[slice_index];

    // The first index on the lead's axis where there are any existing valid sites
    auto const lead_start = [&]{
        for (slice[spec.axis] = start; slice[spec.axis] != end; slice[spec.axis] += step) {
            for (auto& site : slice) {
                if (site.is_valid()) {
                    return slice[spec.axis];
                }
            }
        }
        return slice[spec.axis];
    }();

    if (lead_start == end) {
        throw std::runtime_error("Can't attach lead: completely misses main structure");
    }

    slice_index[spec.axis] = lead_start;
    return slice_index;
}

Junction::Junction(Foundation const& foundation, Spec const& spec)
    : slice_index(attachment_slice(foundation, spec)) {
    // The lead's shape.contains() should be invoked in the center of the shape slice
    auto si3d = shape_slice(foundation, spec.shape);
    auto const si = si3d[spec.axis];
    si3d[spec.axis] = (si.start + si.end) / 2;
    auto const slice = foundation[si3d];
    is_valid = spec.shape.contains(slice.positions());

    if (none_of(is_valid)) {
        throw std::runtime_error("Can't attach lead: no sites in lead junction");
    }
}

} // namespace detail

}} // namespace cpb::leads
