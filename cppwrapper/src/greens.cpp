#include "greens/KPM.hpp"

#include "python_support.hpp"

#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/make_constructor.hpp>

using namespace boost::python;

void export_greens() {
    using tbm::BaseGreens;
    class_<BaseGreens, noncopyable>{"Greens", no_init}
    .def("report", &BaseGreens::report, args("self", "shortform"_kw=false))
    .def("calc_greens", &BaseGreens::calc_greens, args("self", "i", "j", "energy", "broadening"))
    .def("calc_ldos", &BaseGreens::calc_ldos,
         args("self", "energy", "broadening", "position", "sublattice"_kw=-1))
    .def("deferred_ldos", &BaseGreens::deferred_ldos,
         args("self", "energy", "broadening", "position", "sublattice"_kw=-1))
    .add_property("model", internal_ref(&BaseGreens::get_model), &BaseGreens::set_model)
    .add_property("system", &BaseGreens::system)
    ;

    using tbm::Greens;
    using tbm::KPM;
    auto const kpm_defaults = tbm::KPMConfig();
    class_<Greens<KPM>, bases<BaseGreens>, noncopyable>{"KPM", no_init}
    .def("__init__", make_constructor([](tbm::Model const& model, float lambda,
                                         std::pair<float, float> energy, int opt, float lanczos) {
             tbm::KPMConfig config;
             config.lambda = lambda;
             config.min_energy = energy.first;
             config.max_energy = energy.second;
             config.optimization_level = opt;
             config.lanczos_precision = lanczos;

             return new Greens<KPM>(model, config);
         },
         default_call_policies(),
         args("model", "lambda_value"_kw = kpm_defaults.lambda,
              "energy_range"_kw = make_tuple(kpm_defaults.min_energy, kpm_defaults.max_energy),
              "optimization_level"_kw = kpm_defaults.optimization_level,
              "lanczos_precision"_kw = kpm_defaults.lanczos_precision)
         )
    )
    ;
}
