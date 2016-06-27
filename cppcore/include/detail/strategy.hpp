#pragma once
#include "hamiltonian/Hamiltonian.hpp"
#include "support/cppfuture.hpp"

namespace cpb { namespace detail {

/**
 Function object which creates a new Strategy with the appropriate scalar type for the given Model

 BaseStrategy is the abstract base, to which a pointer will be returned.
 Strategy<scalar_t> must be instantiable with float/double and std::complex<float/double>.
 */
template<class BaseStrategy, template<class> class Strategy>
class MakeStrategy {
    static_assert(std::is_base_of<BaseStrategy, Strategy<float>>::value, "");
    using Config = typename Strategy<float>::Config;
    Config config;

public:
    explicit MakeStrategy(Config const& config) : config(config) {}

    std::unique_ptr<BaseStrategy> operator()(Hamiltonian const& h) const {
        std::unique_ptr<BaseStrategy> strategy;

        if (!strategy) strategy = try_strategy<float>(h);
        if (!strategy) strategy = try_strategy<std::complex<float>>(h);
        if (!strategy) strategy = try_strategy<double>(h);
        if (!strategy) strategy = try_strategy<std::complex<double>>(h);
        if (!strategy) {
            throw std::runtime_error("MakeStrategy: unknown Hamiltonian type.");
        }

        return strategy;
    }

private:
    template<class scalar_t>
    std::unique_ptr<BaseStrategy> try_strategy(Hamiltonian const& h) const {
        if (ham::is<scalar_t>(h)) {
            return std::unique_ptr<BaseStrategy>{
                std14::make_unique<Strategy<scalar_t>>(ham::get_shared_ptr<scalar_t>(h), config)
            };
        }
        return {};
    }
};

}} // namespace cpb::detail
