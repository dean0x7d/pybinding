#include "system/Lead.hpp"
#include "system/Foundation.hpp"

namespace tbm { namespace lead {

SliceIndex3D slice_index(Foundation const& foundation, Lead const& lead) {
    auto const size = foundation.get_size();
    auto const foundation_bounds = foundation.get_bounds();
    auto const lead_bounds = detail::find_bounds(lead.shape, foundation.get_lattice());

    auto slice_index = SliceIndex3D();
    for (auto i = 0; i < slice_index.ndims(); ++i) {
        auto const lead_start = lead_bounds.first[i] - foundation_bounds.first[i];
        auto const lead_end = (lead_bounds.second[i] + 1) - foundation_bounds.first[i];
        slice_index[i] = {std::max(lead_start, 0), std::min(lead_end, size[i])};
    }
    return slice_index;
}

} // namespace lead

Lead::Lead(int direction, Shape const& shape) : direction(direction), shape(shape) {
    if (direction == 0 || abs(direction) > 3) {
        throw std::logic_error("Lead direction must be one of: : 1, 2, 3, -1, -2, -3");
    }
}

void attach(Foundation& foundation, Lead const& lead) {
    auto const size = foundation.get_size();
    auto const axis = abs(lead.direction) - 1;
    auto const start = (lead.direction > 0) ? 0 : size[axis] - 1;
    auto const end = (lead.direction > 0) ? size[axis] : -1;
    auto const step = lead.direction / abs(lead.direction);

    auto slice = foundation[lead::slice_index(foundation, lead)];

    // The first index on the lead's axis where there are any existing valid sites
    auto const lead_start = [&]{
        for (slice[axis] = start; slice[axis] != end; slice[axis] += step) {
            for (auto& site : slice) {
                if (site.is_valid()) {
                    return slice[axis];
                }
            }
        }
        return slice[axis];
    }();

    if (lead_start == end) {
        throw std::runtime_error("Can't attach lead: completely misses main structure");
    }

    // Use the shape to determine which slice sites are within the lead
    auto within_lead = [&]{
        slice[axis] = lead_start;
        auto positions = CartesianArray(slice.size());
        for (auto& site : slice) {
            positions[site.get_slice_idx()] = site.get_position();
        }
        return lead.shape.contains(positions);
    }();

    // Fill in the foundation until all lead sites can be connected to foundation sites
    for (slice[axis] = lead_start; slice[axis] != end; slice[axis] += step) {
        for (auto& site : slice) {
            if (!within_lead[site.get_slice_idx()])
                continue;

            if (!site.is_valid()) {
                // Empty sites should be filled in for the lead
                site.set_valid(true);
            } else {
                // Stop once we hit an existing site on the lead's path
                within_lead[site.get_slice_idx()] = false;
            }
        }

        if (none_of(within_lead)) {
            break;
        }
    }

    if (slice[axis] == end) {
        throw std::runtime_error("Can't attach lead: partially misses main structure");
    }
}

} // namespace tbm
