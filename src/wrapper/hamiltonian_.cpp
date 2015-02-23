#include "hamiltonian/Hamiltonian.hpp"
#include "hamiltonian/HamiltonianModifiers.hpp"

#include <boost/python/class.hpp>
#include <boost/python/pure_virtual.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include "python_support.hpp"
using namespace boost::python;

class PyOnsite : public tbm::OnsiteModifier, public wrapper<tbm::OnsiteModifier> {
public:
    virtual bool is_complex() const final {
        if (auto f = get_override("is_complex")) {
            return f();
        }
        return tbm::OnsiteModifier::is_complex();
    }
    
    template<typename Array>
    void apply_in_python(Array& potential, const CartesianArray& position) const {
        object result = get_override("apply")(potential, position.x, position.y, position.z);
        potential = extract<Array>(result);
    }
    
    virtual void apply(ArrayXf& v, const CartesianArray& p) const final { apply_in_python(v, p); }
    virtual void apply(ArrayXcf& v, const CartesianArray& p) const final { apply_in_python(v, p); }
    virtual void apply(ArrayXd& v, const CartesianArray& p) const final { apply_in_python(v, p); }
    virtual void apply(ArrayXcd& v, const CartesianArray& p) const final { apply_in_python(v, p); }
    void apply_dummy(ArrayXf&, const ArrayXf&, const ArrayXf&, const ArrayXf&) const {}
};

class PyHopping : public tbm::HoppingModifier, public wrapper<tbm::HoppingModifier> {
public:
    virtual bool is_complex() const final {
        if (auto f = get_override("is_complex")) {
            return f();
        }
        return tbm::HoppingModifier::is_complex();
    }

    template<typename Array>
    void apply_in_python(Array& hopping, const CartesianArray& pos1, const CartesianArray& pos2) const {
        object o = get_override("apply")(hopping, pos1.x, pos1.y, pos1.z, pos2.x, pos2.y, pos2.z);
        hopping = extract<Array>(o);
    }
    
    using CA = CartesianArray;
    void apply(ArrayXf& h, const CA& p1, const CA& p2) const final { apply_in_python(h, p1, p2); }
    void apply(ArrayXcf& h, const CA& p1, const CA& p2) const final { apply_in_python(h, p1, p2); }
    void apply(ArrayXd& h, const CA& p1, const CA& p2) const final { apply_in_python(h, p1, p2); }
    void apply(ArrayXcd& h, const CA& p1, const CA& p2) const final { apply_in_python(h, p1, p2); }
    void apply_dummy(ArrayXf&, const ArrayXf&, const ArrayXf&, const ArrayXf&,
                     const ArrayXf&, const ArrayXf&, const ArrayXf&) const {}
};

void export_modifiers()
{
    using tbm::Hamiltonian;

    class_<Hamiltonian, std::shared_ptr<Hamiltonian>, noncopyable>{"Hamiltonian", no_init}
    .add_property("_matrix", &Hamiltonian::matrix_union)
    .def_readonly("report", &Hamiltonian::report)
    ;
    register_ptr_to_python<std::shared_ptr<const Hamiltonian>>();

    class_<PyOnsite, noncopyable>{"OnsiteModifier"}
    .def("is_complex", &PyOnsite::is_complex)
    .def("apply", pure_virtual(&PyOnsite::apply_dummy), args("self", "potential", "x", "y", "z"))
    ;
    
    class_<PyHopping, noncopyable>{"HoppingModifier"}
    .def("is_complex", &PyHopping::is_complex)
    .def("apply", pure_virtual(&PyHopping::apply_dummy), args("self", "hopping", "x1", "y1", "z1", "x2", "y2", "z2"))
    ;
}
