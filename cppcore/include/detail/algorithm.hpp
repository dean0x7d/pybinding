#pragma once
#include "detail/config.hpp"

#include <algorithm>

namespace cpb {

/// Loop over each element of `Vector` and call:
///     fill_buffer(reference element, idx_t buffer_position)
/// After every `buffer_capacity` iterations, the `notify_filled` function is called:
///     notify_filled(idx_t last_buffer_size)
/// Following that `process_buffer` is called for each `last_buffer_size` elements:
///     process_buffer(reference element, idx_t buffer_position)
/// Requires `Vector` to be random-access-iterable.
template<class Vector, class F1, class F2, class F3, class F4>
void buffered_for_each(Vector const& v, idx_t buffer_capacity, F1 fill_buffer,
                       F2 notify_filled, F3 process_buffer, F4 notify_processed) {
    using std::begin; using std::end;

    auto it = begin(v);
    auto const stop = end(v);
    auto previous_start = it;
    auto buffer_level = idx_t{0};

    for (; it != stop; ++it) {
        if (buffer_level == buffer_capacity) {
            notify_filled(buffer_capacity);
            for (auto rewound_it = previous_start; rewound_it != it; ++rewound_it) {
                process_buffer(*rewound_it, rewound_it - previous_start);
            }
            previous_start = it;
            buffer_level = 0;
            notify_processed();
        }

        fill_buffer(*it, buffer_level++);
    }

    notify_filled(buffer_level);
    for (auto rewound_it = previous_start; rewound_it != stop; ++rewound_it) {
        process_buffer(*rewound_it, rewound_it - previous_start);
    }
}

} // namespace cpb
