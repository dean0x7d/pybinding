#include "hamiltonian/Hamiltonian.hpp"
#include "hamiltonian/HamiltonianModifiers.hpp"

#include "eigen3_converters.hpp"
#include "python_support.hpp"

#include <boost/python/class.hpp>

using namespace boost::python;
using namespace tbm;

class PyOnsiteModifier : public OnsiteModifier,
                         public wrapper<OnsiteModifier> {
public:
    PyOnsiteModifier(object py_apply, bool is_complex = false, bool is_double = false)
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
    PyHoppingModifier(object py_apply, bool is_complex = false, bool is_double = false)
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
    class_<Hamiltonian, std::shared_ptr<Hamiltonian>, noncopyable>{
        "Hamiltonian", no_init
    }
    .add_property("matrix", internal_ref(&Hamiltonian::matrix_union))
    ;

    class_<HopIdRef>{"HopIdRef", no_init}
    .add_property("ids", internal_ref([](HopIdRef const& s) { return arrayref(s.ids); }))
    .add_property("name_map", copy_value([](HopIdRef const& s) { return s.name_map; }))
    ;

    class_<PyOnsiteModifier, noncopyable>{"OnsiteModifier", init<object, optional<bool, bool>>()}
    .def_readwrite("is_complex", &PyOnsiteModifier::is_complex)
    .def_readwrite("is_double", &PyOnsiteModifier::is_double)
    ;
    class_<PyHoppingModifier, noncopyable>{"HoppingModifier", init<object, optional<bool, bool>>()}
    .def_readwrite("is_complex", &PyHoppingModifier::is_complex)
    .def_readwrite("is_double", &PyHoppingModifier::is_double)
    ;
}
