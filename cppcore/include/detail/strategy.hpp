#pragma once
#include "hamiltonian/Hamiltonian.hpp"
#include "support/cppfuture.hpp"

namespace tbm {

class Model;

namespace detail {

/**
 Function object which creates a new Strategy with the appropriate scalar type for the given Model

 BaseStrategy is the abstract base, to which a pointer will be returned.
 Strategy<scalar_t> must be instantiable with float/double and std::complex<float/double>.
 */
template<class BaseStrategy, template<class> class Strategy>
class MakeStrategy {
    static_assert(std::is_base_of<BaseStrategy, Strategy<float>>::value, "");
    using Config = typename Strategy<float>::Config;

public:
    explicit MakeStrategy(Config const& config) : config(config) {}

    std::unique_ptr<BaseStrategy> operator()(Model const& model) const {
        std::unique_ptr<BaseStrategy> strategy;

        if (!strategy) strategy = try_strategy<float>(model);
        if (!strategy) strategy = try_strategy<std::complex<float>>(model);
        if (!strategy) strategy = try_strategy<double>(model);
        if (!strategy) strategy = try_strategy<std::complex<double>>(model);
        if (!strategy)
            throw std::runtime_error{"MakeStrategy: unknown Hamiltonian type."};

        return strategy;
    }

private:
    template<class scalar_t>
    std::unique_ptr<BaseStrategy> try_strategy(Model const& model) const {
        auto const& h = model.hamiltonian();
        if (ham::is<scalar_t>(h)) {
            auto strategy = std14::make_unique<Strategy<scalar_t>>(config);
            strategy->set_hamiltonian(h);
            return std::move(strategy);
        }
        return {};
    }

private:
    Config config;
};

}} // namespace tbm::detail
