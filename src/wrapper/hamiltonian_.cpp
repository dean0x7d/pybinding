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
public:
    virtual bool is_complex() const final {
        if (auto f = get_override("is_complex")) {
            return f();
        }
        return tbm::HoppingModifierImpl::is_complex();
    }

    template<typename Array>
    void apply_in_python(Array& hopping, const CartesianArray& pos1, const CartesianArray& pos2) const {
        object result = get_override("apply")(
            DenseURef{hopping},
            DenseURef{pos1.x}, DenseURef{pos1.y}, DenseURef{pos1.z},
            DenseURef{pos2.x}, DenseURef{pos2.y}, DenseURef{pos2.z}
        );
        extract_array(hopping, result);
    }
    
    using CA = CartesianArray;
    void apply(ArrayXf& h, const CA& p1, const CA& p2) const final { apply_in_python(h, p1, p2); }
    void apply(ArrayXcf& h, const CA& p1, const CA& p2) const final { apply_in_python(h, p1, p2); }
    void apply(ArrayXd& h, const CA& p1, const CA& p2) const final { apply_in_python(h, p1, p2); }
    void apply(ArrayXcd& h, const CA& p1, const CA& p2) const final { apply_in_python(h, p1, p2); }
    void apply_dummy(ArrayXf&, const ArrayXf&, const ArrayXf&, const ArrayXf&,
                     const ArrayXf&, const ArrayXf&, const ArrayXf&) const {}
};

void export_modifiers() {
    using tbm::Hamiltonian;
    class_<Hamiltonian, noncopyable>{"Hamiltonian", no_init}
    .add_property("matrix", internal_ref(&Hamiltonian::matrix_union))
    .def_readonly("report", &Hamiltonian::report)
    ;

    class_<PyHopping, noncopyable>{"HoppingModifier"}
    .def("is_complex", &PyHopping::is_complex)
    .def("apply", pure_virtual(&PyHopping::apply_dummy), args("self", "hopping", "x1", "y1", "z1", "x2", "y2", "z2"))
    ;
    class_<PyOnsite, noncopyable>{"OnsiteModifier"};
}
