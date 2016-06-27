#include "hamiltonian/Hamiltonian.hpp"

#include "eigen3_converters.hpp"
#include "python_support.hpp"

#include <boost/python/class.hpp>

using namespace boost::python;
using namespace cpb;

class PyOnsiteModifier : public OnsiteModifier,
                         public wrapper<OnsiteModifier> {
public:
    PyOnsiteModifier(object py_apply, bool is_complex, bool is_double)
        : OnsiteModifier({}, is_complex, is_double) {
        apply = [py_apply](ComplexArrayRef energy, CartesianArray const& p, SubIdRef sub) {
            object result = py_apply(energy, arrayref(p.x), arrayref(p.y), arrayref(p.z), sub);
            num::match<ArrayX>(energy, ExtractArray{result});
        };
    }
};

class PyHoppingModifier : public HoppingModifier,
                          public wrapper<HoppingModifier> {
public:
    PyHoppingModifier(object py_apply, bool is_complex, bool is_double)
        : HoppingModifier({}, is_complex, is_double) {
        apply = [py_apply](ComplexArrayRef energy, CartesianArray const& p1,
                           CartesianArray const& p2, HopIdRef hopping) {
            object result = py_apply(energy, arrayref(p1.x), arrayref(p1.y), arrayref(p1.z),
                                     arrayref(p2.x), arrayref(p2.y), arrayref(p2.z), hopping);
            num::match<ArrayX>(energy, ExtractArray{result});
        };
    }
};

void export_modifiers() {
    class_<Hamiltonian>{"Hamiltonian", no_init}
    .add_property("csrref", return_internal_copy(&Hamiltonian::csrref))
    ;

    class_<HopIdRef>{"HopIdRef", no_init}
    .add_property("ids", return_internal_copy([](HopIdRef const& s) { return arrayref(s.ids); }))
    .add_property("name_map", return_copy([](HopIdRef const& s) { return s.name_map; }))
    ;

    class_<PyOnsiteModifier, noncopyable>{
        "OnsiteModifier", init<object, bool, bool>(
            args("self", "apply", "is_complex"_kw=false, "is_double"_kw=false)
        )
    }
    .def_readwrite("is_complex", &PyOnsiteModifier::is_complex)
    .def_readwrite("is_double", &PyOnsiteModifier::is_double)
    ;

    class_<PyHoppingModifier, noncopyable>{
        "HoppingModifier", init<object, bool, bool>(
            args("self", "apply", "is_complex"_kw=false, "is_double"_kw=false)
        )
    }
    .def_readwrite("is_complex", &PyHoppingModifier::is_complex)
    .def_readwrite("is_double", &PyHoppingModifier::is_double)
    ;
}
