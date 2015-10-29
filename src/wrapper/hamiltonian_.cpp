#include "hamiltonian/Hamiltonian.hpp"
#include "hamiltonian/HamiltonianModifiers.hpp"

#include "converters/eigen3.hpp"
#include "python_support.hpp"

#include <boost/python/class.hpp>
#include <boost/python/pure_virtual.hpp>
#include <boost/python/tuple.hpp>

using namespace boost::python;

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
    
    template<typename Array>
    void apply_(Array& potential, CA p, SA s) const {
        object result = get_override("apply")(
            DenseURef{potential}, DenseURef{p.x}, DenseURef{p.y}, DenseURef{p.z}, DenseURef{s}
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

    template<typename Array>
    void apply_(Array& hopping, HA id, CA p1, CA p2) const {
        object result = get_override("apply")(
            DenseURef{hopping}, DenseURef{id},
            DenseURef{p1.x}, DenseURef{p1.y}, DenseURef{p1.z},
            DenseURef{p2.x}, DenseURef{p2.y}, DenseURef{p2.z}
        );
        extract_array(hopping, result);
    }
    
    virtual void apply(ArrayXf& h, HA id, CA p1, CA p2) const final { apply_(h, id, p1, p2); }
    virtual void apply(ArrayXd& h, HA id, CA p1, CA p2) const final { apply_(h, id, p1, p2); }
    virtual void apply(ArrayXcf& h, HA id, CA p1, CA p2) const final { apply_(h, id, p1, p2); }
    virtual void apply(ArrayXcd& h, HA id, CA p1, CA p2) const final { apply_(h, id, p1, p2); }
};

void export_modifiers() {
    using tbm::Hamiltonian;
    class_<Hamiltonian, noncopyable>{"Hamiltonian", no_init}
    .add_property("matrix", internal_ref(&Hamiltonian::matrix_union))
    .def_readonly("report", &Hamiltonian::report)
    ;

    class_<PyOnsite, noncopyable>{"OnsiteModifier"};
    class_<PyHopping, noncopyable>{"HoppingModifier"};
}
