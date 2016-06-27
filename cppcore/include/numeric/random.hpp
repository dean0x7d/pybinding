#pragma once
#include "numeric/dense.hpp"
#include "support/cppfuture.hpp"

#include <random>

namespace cpb { namespace num {

namespace detail {
    template<class Container>
    using get_element_t = get_real_t<std14::decay_t<decltype(std::declval<Container>()[0])>>;

    template<class scalar_t>
    using select_distribution = std14::conditional_t<
        std::is_floating_point<scalar_t>::value,
        std::uniform_real_distribution<scalar_t>,
        std::uniform_int_distribution<scalar_t>
    >;
}

/*
 Fill the container with uniformly distributed random data
 */
template<class Container>
void random_fill(Container& container, std::uint_fast32_t seed = std::mt19937::default_seed) {
    using scalar_t = detail::get_element_t<Container>;
    static_assert(std::is_arithmetic<scalar_t>::value, "");

    auto generator = std::mt19937(seed);
    auto distribution = detail::select_distribution<scalar_t>();

    for (auto& value : container) {
        value = distribution(generator);
    }
}

/*
 Initialize `Container` with `args` and fill with random data uniformly distributed
 on the interval [0, 1) for real numbers or [0, int_max] for integers
 */
template<class Container, class... Args>
Container make_random(Args&&... args) {
    Container container(std::forward<Args>(args)...);
    random_fill(container);
    return container;
};

}} // namespace cpb::num
