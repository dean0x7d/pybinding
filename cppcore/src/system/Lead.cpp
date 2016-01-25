#include "system/Lead.hpp"
#include "system/Foundation.hpp"

namespace tbm { namespace lead {

SliceIndex3D passthrough_slice(Foundation const& foundation, Lead const& lead) {
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

SliceIndex3D attachment_slice(Foundation const& foundation, Lead const& lead) {
    auto const size = foundation.get_size();
    auto const start = (lead.sign > 0) ? 0 : size[lead.axis] - 1;
    auto const end = (lead.sign > 0) ? size[lead.axis] : -1;

    auto slice_index = lead::passthrough_slice(foundation, lead);
    auto slice = foundation[slice_index];

    // The first index on the lead's axis where there are any existing valid sites
    auto const lead_start = [&]{
        for (slice[lead.axis] = start; slice[lead.axis] != end; slice[lead.axis] += lead.sign) {
            for (auto& site : slice) {
                if (site.is_valid()) {
                    return slice[lead.axis];
                }
            }
        }
        return slice[lead.axis];
    }();

    if (lead_start == end) {
        throw std::runtime_error("Can't attach lead: completely misses main structure");
    }

    slice_index[lead.axis] = lead_start;
    return slice_index;
}

} // namespace lead

Lead::Lead(int direction, Shape const& shape)
    : axis(abs(direction) - 1),
      sign(direction != 0 ? direction / abs(direction) : 0),
      shape(shape) {
    if (direction == 0 || abs(direction) > 3) {
        throw std::logic_error("Lead direction must be one of: 1, 2, 3, -1, -2, -3");
    }
}

void attach(Foundation& foundation, Lead const& lead) {
    auto const size = foundation.get_size();
    auto const end = (lead.sign > 0) ? size[lead.axis] : -1;

    auto slice = foundation[lead::attachment_slice(foundation, lead)];
    auto within_lead = lead.shape.contains(slice.positions());

    // Fill in the foundation until all lead sites can be connected to foundation sites
    for (; slice[lead.axis] != end; slice[lead.axis] += lead.sign) {
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

    if (slice[lead.axis] == end) {
        throw std::runtime_error("Can't attach lead: partially misses main structure");
    }
}

} // namespace tbm
