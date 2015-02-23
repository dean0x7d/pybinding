#include "result/DOS.hpp"
#include "result/LDOSpoint.hpp"
#include "result/LDOSenergy.hpp"

#include <boost/python/class.hpp>
#include <boost/python/list.hpp>
#include "python_support.hpp"
#include "converters/tuple.hpp"
using namespace boost::python;

void export_results()
{
    using tbm::Result;
    using tbm::DOS;
    using tbm::LDOSpoint;
    using tbm::LDOSenergy;

    create_vector_converter<Cartesian>();

    class_<Result>{"Results", no_init};

    class_<DOS, bases<Result>>{
        "DOS", "Save DOS as a function of energy.",
        init<ArrayXd, float> {args("self", "energy_range", "broadening")}
    }
    .add_property("dos", const_ref(&DOS::get_dos))
    .add_property("energy", const_ref(&DOS::get_energy))
    ;
    
    class_<LDOSpoint, bases<Result>>{
        "LDOSpoint", "Calculate the LDOS at the given point as a function of energy.",
        init<ArrayXd, float, Cartesian, int, std::vector<Cartesian>>{
            args("self", "energy", "broadening", "position", "sublattice"_a=-1, "k_path"_a=list{})
        }
    }
    .add_property("ldos", const_ref(&LDOSpoint::get_ldos))
    .add_property("energy", const_ref(&LDOSpoint::get_energy))
    ;
    
    class_<LDOSenergy, bases<Result>>{
        "LDOSenergy", "Calculate the LDOS at the given energy as a function of position.",
        init<float, float, int>{args("self", "energy", "broadening", "sublattice"_a=-1)}
    }
    .add_property("ldos", const_ref(&LDOSenergy::get_ldos))
    ;
}
