#include "hamiltonian/Hamiltonian.hpp"
#include "hamiltonian/HamiltonianModifiers.hpp"

#include "eigen3_converters.hpp"
#include "python_support.hpp"

#include <boost/python/class.hpp>
#include <boost/python/pure_virtual.hpp>
#include <boost/python/tuple.hpp>

using namespace boost::python;
using namespace tbm;

class PyOnsite : public OnsiteModifierImpl,
                 public wrapper<OnsiteModifierImpl> {
public:
    virtual bool is_complex() const final {
        if (auto f = get_override("is_complex")) {
            return f();
        }
        return tbm::OnsiteModifierImpl::is_complex();
    }
    
    template<class Array>
    void apply_(Array& potential, CartesianArray const& p, SubIdRef s) const {
        object result = get_override("apply")(
            arrayref(potential), arrayref(p.x), arrayref(p.y), arrayref(p.z), s
        );
        extract_array(potential, result);
    }
    
    void apply(ArrayXf& v, CartesianArray const& p, SubIdRef s) const final {apply_(v, p, s); }
    void apply(ArrayXcf& v, CartesianArray const& p, SubIdRef s) const final {apply_(v, p, s); }
    void apply(ArrayXd& v, CartesianArray const& p, SubIdRef s) const final {apply_(v, p, s); }
    void apply(ArrayXcd& v, CartesianArray const& p, SubIdRef s) const final {apply_(v, p, s); }
};

class PyHopping : public tbm::HoppingModifierImpl,
                  public wrapper<tbm::HoppingModifierImpl> {
    using CA = CartesianArray const&;

public:
    virtual bool is_complex() const final {
        if (auto f = get_override("is_complex")) {
            return f();
        }
        return tbm::HoppingModifierImpl::is_complex();
    }

    template<class Array>
    void apply_(Array& energy, CA p1, CA p2, HopIdRef hopping) const {
        object result = get_override("apply")(
            arrayref(energy),
            arrayref(p1.x), arrayref(p1.y), arrayref(p1.z),
            arrayref(p2.x), arrayref(p2.y), arrayref(p2.z),
            hopping
        );
        extract_array(energy, result);
    }
    
    void apply(ArrayXf& e, CA p1, CA p2, HopIdRef h) const final { apply_(e, p1, p2, h); }
    void apply(ArrayXd& e, CA p1, CA p2, HopIdRef h) const final { apply_(e, p1, p2, h); }
    void apply(ArrayXcf& e, CA p1, CA p2, HopIdRef h) const final { apply_(e, p1, p2, h); }
    void apply(ArrayXcd& e, CA p1, CA p2, HopIdRef h) const final { apply_(e, p1, p2, h); }
};

void export_modifiers() {
    using tbm::Hamiltonian;
    class_<Hamiltonian, std::shared_ptr<Hamiltonian>, noncopyable>{"Hamiltonian", no_init}
    .add_property("matrix", internal_ref(&Hamiltonian::matrix_union))
    ;

    class_<HopIdRef>{"HopIdRef", no_init}
    .add_property("ids", internal_ref([](HopIdRef const& s) { return arrayref(s.ids); }))
    .add_property("name_map", copy_value([](HopIdRef const& s) { return s.name_map; }))
    ;

    class_<PyOnsite, noncopyable>{"OnsiteModifier"}
    .def_readwrite("is_double", &PyOnsite::is_double);
    class_<PyHopping, noncopyable>{"HoppingModifier"}
    .def_readwrite("is_double", &PyHopping::is_double);
}
