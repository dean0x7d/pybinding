#include "greens/KPM.hpp"

#include <boost/python/class.hpp>
using namespace boost::python;

void export_greens()
{
    using tbm::Greens;
    using tbm::GreensFactory;
    using tbm::KPMFactory;

    class_<Greens, noncopyable> {
        "Greens", no_init
    };
    class_<GreensFactory, noncopyable> {
        "GreensFactory", no_init
    };

    class_<KPMFactory, bases<GreensFactory>, noncopyable> {
        "KPM", "Green's function via kernel polynomial method.",
        init<double, double, double> {
            (arg("self"),
             arg("lambda_value") = KPMFactory::defaults::lambda,
             arg("e_min") = KPMFactory::defaults::min_energy,
             arg("e_max") = KPMFactory::defaults::max_energy)
        }
    }
    .def("advanced", &KPMFactory::advanced,
         (arg("self"),
          arg("use_reordering") = KPMFactory::defaults::use_reordering,
          arg("lanczos_precision") = KPMFactory::defaults::lanczos_precision,
          arg("scaling_tolerance") = KPMFactory::defaults::scaling_tolerance)
    );
}
