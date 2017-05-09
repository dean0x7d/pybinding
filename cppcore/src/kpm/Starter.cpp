#include "kpm/Starter.hpp"

#include "numeric/random.hpp"

namespace cpb { namespace kpm {

namespace {

struct ConstantStarter {
    OptimizedHamiltonian const& oh;
    VectorXcd const& alpha;

    ConstantStarter(OptimizedHamiltonian const& oh, VectorXcd const& alpha)
        : oh(oh), alpha(alpha) {}

    var::complex<VectorX> operator()(var::scalar_tag tag) const { return tag.match(*this); }

    template<class scalar_t>
    var::complex<VectorX> operator()(var::tag<scalar_t>) const {
        auto r0 = num::force_cast<scalar_t>(alpha);
        oh.reorder(r0); // needed to maintain consistent results for all optimizations
        return r0;
    }
};

struct UnitStarter {
    idx_t size;
    ArrayXi sources;
    idx_t i = 0;

    UnitStarter(OptimizedHamiltonian const& oh) : size(oh.size()), sources(oh.idx().src) {}

    var::complex<VectorX> operator()(var::scalar_tag tag) {
        return var::apply_visitor(*this, tag);
    }

    template<class scalar_t>
    var::complex<VectorX> operator()(var::tag<scalar_t>) {
        auto r0 = VectorX<scalar_t>::Zero(size).eval();
        if (i < sources.size()) {
            r0[sources[i]] = 1;
            ++i;
        }
        return r0;
    }
};

struct RandomStarter {
    OptimizedHamiltonian const& oh;
    VariantCSR op;
    std::mt19937 generator;

    RandomStarter(OptimizedHamiltonian const& oh, VariantCSR const& op) : oh(oh), op(op) {}

    var::complex<VectorX> operator()(var::scalar_tag tag) {
        return var::apply_visitor(*this, tag);
    }

    template<class real_t>
    var::complex<VectorX> operator()(var::tag<real_t>) {
        auto r0 = transform<VectorX>(
            num::make_random<VectorX<real_t>>(oh.size(), generator),
            [](real_t x) -> real_t { return (x < 0.5f) ? -1.f : 1.f; }
        );

        if (op) { r0 = op.get<real_t>() * r0; }

        oh.reorder(r0); // needed to maintain consistent results for all optimizations
        return r0;
    }

    template<class real_t>
    var::complex<VectorX> operator()(var::tag<std::complex<real_t>>) {
        auto const phase = num::make_random<ArrayX<real_t>>(oh.size(), generator);
        auto const k = std::complex<real_t>{2 * constant::pi * constant::i1};
        auto r0 = exp(k * phase).matrix().eval();

        if (op) { r0 = op.get<std::complex<real_t>>() * r0; }

        oh.reorder(r0);
        return r0;
    }
};

} // anonymous namespace

Starter constant_starter(OptimizedHamiltonian const& oh, VectorXcd const& alpha) {
    return {ConstantStarter(oh, alpha), oh.size()};
}

Starter unit_starter(OptimizedHamiltonian const& oh) {
    return {UnitStarter(oh), oh.size()};
}

Starter random_starter(OptimizedHamiltonian const& oh, VariantCSR const& op) {
    return {RandomStarter(oh, op), oh.size()};
}

}} // namespace cpb::kpm
