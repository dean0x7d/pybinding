#include "hamiltonian/Hamiltonian.hpp"
#include "hamiltonian/HamiltonianModifiers.hpp"

#include "eigen3_converters.hpp"
#include "python_support.hpp"

#include <boost/python/class.hpp>
#include <boost/python/pure_virtual.hpp>
#include <boost/python/tuple.hpp>

using namespace boost::python;
using namespace tbm;

class PyOnsite : public tbm::OnsiteModifierImpl,
                 public wrapper<tbm::OnsiteModifierImpl> {
    using CA = CartesianArray const&;
    using SA = ArrayX<tbm::sub_id> const&;

public:
    virtual bool is_complex() const final {
        if (auto f = get_override("is_complex")) {
            return f();
        }
        return tbm::OnsiteModifierImpl::is_complex();
    }
    
    template<class Array>
    void apply_(Array& potential, CA p, SA s) const {
        object result = get_override("apply")(
            arrayref(potential), arrayref(p.x), arrayref(p.y), arrayref(p.z), arrayref(s)
        );
        extract_array(potential, result);
    }
    
    virtual void apply(ArrayXf& v, CA p, SA s) const final {apply_(v, p, s); }
    virtual void apply(ArrayXcf& v, CA p, SA s) const final {apply_(v, p, s); }
    virtual void apply(ArrayXd& v, CA p, SA s) const final {apply_(v, p, s); }
    virtual void apply(ArrayXcd& v, CA p, SA s) const final {apply_(v, p, s); }
};

class PyHopping : public tbm::HoppingModifierImpl,
                  public wrapper<tbm::HoppingModifierImpl> {
    using CA = CartesianArray const&;
    using HA = ArrayX<tbm::hop_id> const&;

public:
    virtual bool is_complex() const final {
        if (auto f = get_override("is_complex")) {
            return f();
        }
        return tbm::HoppingModifierImpl::is_complex();
    }

    template<class Array>
    void apply_(Array& hopping, CA p1, CA p2, HA id) const {
        object result = get_override("apply")(
            arrayref(hopping),
            arrayref(p1.x), arrayref(p1.y), arrayref(p1.z),
            arrayref(p2.x), arrayref(p2.y), arrayref(p2.z),
            arrayref(id)
        );
        extract_array(hopping, result);
    }
    
    virtual void apply(ArrayXf& h, CA p1, CA p2, HA id) const final { apply_(h, p1, p2, id); }
    virtual void apply(ArrayXd& h, CA p1, CA p2, HA id) const final { apply_(h, p1, p2, id); }
    virtual void apply(ArrayXcf& h, CA p1, CA p2, HA id) const final { apply_(h, p1, p2, id); }
    virtual void apply(ArrayXcd& h, CA p1, CA p2, HA id) const final { apply_(h, p1, p2, id); }
};

void export_modifiers() {
    using tbm::Hamiltonian;
    class_<Hamiltonian, std::shared_ptr<Hamiltonian>, noncopyable>{"Hamiltonian", no_init}
    .add_property("matrix", internal_ref(&Hamiltonian::matrix_union))
    ;

    class_<PyOnsite, noncopyable>{"OnsiteModifier"};
    class_<PyHopping, noncopyable>{"HoppingModifier"};
}
