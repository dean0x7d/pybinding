#include "greens/KPM.hpp"

#include "python_support.hpp"

#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>

using namespace boost::python;


void export_greens() {
    using tbm::Greens;
    using tbm::KPM;

    class_<Greens, noncopyable>{"Greens", no_init}
    .def("report", &Greens::report, args("self", "shortform"_kw=false))
    .def("calc_greens", &Greens::calc_greens, args("self", "i", "j", "energy", "broadening"))
    .def("calc_ldos", &Greens::calc_ldos,
         args("self", "energy", "broadening", "position", "sublattice"_kw=-1))
    .def("deferred_ldos", &Greens::deferred_ldos,
         args("self", "energy", "broadening", "position", "sublattice"_kw=-1))
    .add_property("model", internal_ref(&Greens::get_model), &Greens::set_model)
    .add_property("system", &Greens::system)
    ;

    auto const kpm_defaults = tbm::KPM::Config{};
    class_<KPM, bases<Greens>, noncopyable> {
        "KPM", "Green's function via kernel polynomial method.",
        init<tbm::Model const&, float, std::pair<float, float>, int, float>{args(
            "self", "model",
            "lambda_value"_kw = kpm_defaults.lambda,
            "energy_range"_kw = make_tuple(kpm_defaults.min_energy, kpm_defaults.max_energy),
            "optimization_level"_kw = kpm_defaults.optimization_level,
            "lanczos_precision"_kw = kpm_defaults.lanczos_precision
        )}
    }
    ;
}
