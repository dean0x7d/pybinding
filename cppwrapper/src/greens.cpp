#include "greens/KPM.hpp"

#include "python_support.hpp"

#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/make_constructor.hpp>

using namespace boost::python;
using namespace cpb;

namespace {

using OptHamVariant = var::variant<kpm::OptimizedHamiltonian<float>,
                                   kpm::OptimizedHamiltonian<double>,
                                   kpm::OptimizedHamiltonian<std::complex<float>>,
                                   kpm::OptimizedHamiltonian<std::complex<double>>>;

struct MakeOptHam {
    int i;

    template<class scalar_t>
    OptHamVariant operator()(SparseMatrixRC<scalar_t> const& m) const {
        auto ret = kpm::OptimizedHamiltonian<scalar_t>(m.get(), {kpm::MatrixConfig::Reorder::ON,
                                                                 kpm::MatrixConfig::Format::CSR});
        auto indices = std::vector<int>(m->rows());
        std::iota(indices.begin(), indices.end(), 0);
        auto bounds = kpm::Bounds<scalar_t>(m.get(), KPMConfig{}.lanczos_precision);
        ret.optimize_for({i, indices}, bounds.scaling_factors());
        return ret;
    }
};

struct ReturnMatrix {
    template<class scalar_t>
    ComplexCsrConstRef operator()(kpm::OptimizedHamiltonian<scalar_t> const& oh) const {
        return csrref(oh.csr());
    }
};

struct Sizes {
    template<class scalar_t>
    std::vector<int> const& operator()(kpm::OptimizedHamiltonian<scalar_t> const& oh) const {
        return oh.sizes().get_data();
    }
};

struct Indices {
    template<class scalar_t>
    ArrayXi const& operator()(kpm::OptimizedHamiltonian<scalar_t> const& oh) const {
        return oh.idx().cols;
    }
};

class PyOptHam {
    Hamiltonian h;
    OptHamVariant oh;

public:
    PyOptHam(Hamiltonian const& h, int index)
        : h(h), oh(var::apply_visitor(MakeOptHam{index}, h.get_variant())) {}

    ComplexCsrConstRef matrix() const { return var::apply_visitor(ReturnMatrix{}, oh); }
    std::vector<int> const& sizes() const { return var::apply_visitor(Sizes{}, oh); }
    ArrayXi const& indices() const { return var::apply_visitor(Indices{}, oh); }
};

} // anonymously namespace


void export_greens() {
    class_<PyOptHam>{"OptimizedHamiltonian", init<Hamiltonian const&, int>()}
    .add_property("matrix", return_internal_copy(&PyOptHam::matrix))
    .add_property("sizes", return_arrayref(&PyOptHam::sizes))
    .add_property("indices", return_arrayref(&PyOptHam::indices))
    ;

    class_<BaseGreens, noncopyable>{"Greens", no_init}
    .def("report", &BaseGreens::report, args("self", "shortform"_kw=false))
    .def("calc_greens", &BaseGreens::calc_greens,
         args("self", "row", "col", "energy", "broadening"))
    .def("calc_greens", &BaseGreens::calc_greens_vector,
         args("self", "row", "cols", "energy", "broadening"))
    .def("calc_ldos", &BaseGreens::calc_ldos,
         args("self", "energy", "broadening", "position", "sublattice"_kw=-1))
    .def("deferred_ldos", &BaseGreens::deferred_ldos,
         args("self", "energy", "broadening", "position", "sublattice"_kw=-1))
    .add_property("model", return_reference(&BaseGreens::get_model), &BaseGreens::set_model)
    .add_property("system", &BaseGreens::system)
    ;

    auto const kpm_defaults = KPMConfig();
    class_<Greens<KPM>, bases<BaseGreens>, noncopyable>{"KPM", no_init}
    .def("__init__", make_constructor([](Model const& model, float lambda,
                                         std::pair<float, float> energy, int opt, float lanczos) {
             KPMConfig config;
             config.lambda = lambda;
             config.min_energy = energy.first;
             config.max_energy = energy.second;
             config.opt_level = opt;
             config.lanczos_precision = lanczos;

             return new Greens<KPM>(model, config);
         },
         default_call_policies(),
         args("model", "lambda_value"_kw = kpm_defaults.lambda,
              "energy_range"_kw = make_tuple(kpm_defaults.min_energy, kpm_defaults.max_energy),
              "optimization_level"_kw = kpm_defaults.opt_level,
              "lanczos_precision"_kw = kpm_defaults.lanczos_precision)
         )
    )
    ;

#ifdef CPB_USE_CUDA
    class_<Greens<KPMcuda>, bases<BaseGreens>, noncopyable>{"KPMcuda", no_init}
    .def("__init__", make_constructor([](Model const& model, float lambda,
                                         std::pair<float, float> energy, int opt) {
             KPMConfig config;
             config.lambda = lambda;
             config.min_energy = energy.first;
             config.max_energy = energy.second;
             config.opt_level = opt;
             return new Greens<KPMcuda>(model, config);
         },
         default_call_policies(),
         args("model", "lambda_value"_kw = kpm_defaults.lambda,
              "energy_range"_kw = make_tuple(kpm_defaults.min_energy, kpm_defaults.max_energy),
              "optimization_level"_kw = kpm_defaults.opt_level)
         )
    )
    ;
#endif // CPB_USE_CUDA
}
