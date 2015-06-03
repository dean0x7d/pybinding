#include "greens/KPM.hpp"

#include <boost/python/class.hpp>
using namespace boost::python;

void export_greens() {
    using tbm::Greens;
    using tbm::KPM;

    class_<Greens, noncopyable>{"Greens", no_init}
    .def("set_model", &Greens::set_model, args("self", "model"))
    .def("report", &Greens::report, args("self", "shortform"_kw=false))
    ;

    class_<KPM, bases<Greens>, noncopyable> {
        "KPM", "Green's function via kernel polynomial method.",
        init<const std::shared_ptr<const tbm::Model>&, float, std::pair<float, float>>{args(
            "self", "model",
            "lambda_value"_kw = KPM::defaults.lambda,
            "energy_range"_kw = std::make_pair(KPM::defaults.min_energy, KPM::defaults.max_energy)
        )}
    }
    .def("advanced", &KPM::advanced, args(
        "self",
        "use_reordering"_kw = KPM::defaults.use_reordering,
        "lanczos_precision"_kw = KPM::defaults.lanczos_precision,
        "scaling_tolerance"_kw = KPM::defaults.scaling_tolerance
    ))
    .def("calc_greens", &KPM::calc_greens, args("self", "i", "j", "energy", "broadening"))
    .def("calc_ldos", &KPM::calc_ldos,
         args("self", "energy", "broadening", "position", "sublattice"_kw=-1))
    ;
}
