#include "system/StructureModifiers.hpp"
#include "hamiltonian/HamiltonianModifiers.hpp"
#include "wrappers.hpp"
using namespace cpb;

namespace {

/// Extract an Eigen array from a Python object, but avoid a copy if possible
struct ExtractModifierResult {
    py::object o;

    template<class EigenType>
    void operator()(Eigen::Map<EigenType> eigen_map) const {
        static_assert(EigenType::IsVectorAtCompileTime, "");

        using scalar_t = typename EigenType::Scalar;
        using NumpyType = py::array_t<typename EigenType::Scalar>;
        if (!py::isinstance<NumpyType>(o)) {
            std::is_floating_point<scalar_t>()
                  ? throw ComplexOverride()
                  : throw std::runtime_error("Unexpected modifier result size");
        }

        auto const a = py::reinterpret_borrow<NumpyType>(o);
        if (eigen_map.size() != static_cast<idx_t>(a.size())) {
            throw std::runtime_error("Unexpected modifier result size");
        }

        if (eigen_map.data() != a.data()) {
            eigen_map = Eigen::Map<EigenType const>(a.data(), static_cast<idx_t>(a.size()));
        }
    }
};

template<class EigenType, class T>
void extract_modifier_result(T& v, py::object const& o) {
    ExtractModifierResult{o}(Eigen::Map<EigenType>(v.data(), v.size()));
}

} // anonymous namespace

template<class T>
SiteGenerator* init_site_generator(string_view name, T const& energy, py::object make) {
    auto system_type = py::module::import("pybinding.system").attr("System");
    return new SiteGenerator(
        name, detail::canonical_onsite_energy(energy),
        [make, system_type](System const& s) {
            py::gil_scoped_acquire guard{};
            auto t = make(system_type(&s)).cast<py::tuple>();
            return CartesianArray(t[0].cast<ArrayXf>(),
                                  t[1].cast<ArrayXf>(),
                                  t[2].cast<ArrayXf>());
        }
    );
};

template<class T>
HoppingGenerator* init_hopping_generator(std::string const& name, T const& energy,
                                         py::object make) {
    auto system_type = py::module::import("pybinding.system").attr("System");
    return new HoppingGenerator(
        name, detail::canonical_hopping_energy(energy),
        [make, system_type](System const& s) {
            py::gil_scoped_acquire guard{};
            auto t = make(system_type(&s)).cast<py::tuple>();
            return HoppingGenerator::Result{t[0].cast<ArrayXi>(), t[1].cast<ArrayXi>()};
        }
    );
}

void wrap_modifiers(py::module& m) {
    py::class_<SiteStateModifier>(m, "SiteStateModifier")
        .def(py::init([](py::object apply, int min_neighbors) {
            return new SiteStateModifier(
                [apply](Eigen::Ref<ArrayX<bool>> state, CartesianArrayConstRef p, string_view s) {
                    py::gil_scoped_acquire guard{};
                    auto result = apply(
                        arrayref(state), arrayref(p.x()), arrayref(p.y()), arrayref(p.z()), s
                    );
                    extract_modifier_result<ArrayX<bool>>(state, result);
                },
                min_neighbors
            );
        }), "apply"_a, "min_neighbors"_a=0);

    py::class_<PositionModifier>(m, "PositionModifier")
        .def(py::init([](py::object apply) {
            return new PositionModifier([apply](CartesianArrayRef p, string_view sub) {
                py::gil_scoped_acquire guard{};
                auto t = py::tuple(apply(arrayref(p.x()), arrayref(p.y()), arrayref(p.z()), sub));
                extract_modifier_result<ArrayXf>(p.x(), t[0]);
                extract_modifier_result<ArrayXf>(p.y(), t[1]);
                extract_modifier_result<ArrayXf>(p.z(), t[2]);
            });
        }));

    py::class_<SiteGenerator>(m, "SiteGenerator")
        .def(py::init(&init_site_generator<std::complex<double>>))
        .def(py::init(&init_site_generator<VectorXd>))
        .def(py::init(&init_site_generator<MatrixXcd>));

    py::class_<HoppingGenerator>(m, "HoppingGenerator")
        .def(py::init(&init_hopping_generator<std::complex<double>>))
        .def(py::init(&init_hopping_generator<MatrixXcd>));

    py::class_<OnsiteModifier>(m, "OnsiteModifier")
        .def(py::init([](py::object apply, bool is_complex, bool is_double) {
            return new OnsiteModifier(
                [apply](ComplexArrayRef energy, CartesianArrayConstRef p, string_view sublattice) {
                    py::gil_scoped_acquire guard{};
                    auto result = apply(
                        energy, arrayref(p.x()), arrayref(p.y()), arrayref(p.z()), sublattice
                    );
                    num::match<ArrayX>(energy, ExtractModifierResult{result});
                },
                is_complex, is_double
            );
        }), "apply"_a, "is_complex"_a=false, "is_double"_a=false)
        .def_readwrite("is_complex", &OnsiteModifier::is_complex)
        .def_readwrite("is_double", &OnsiteModifier::is_double);

    py::class_<HoppingModifier>(m, "HoppingModifier")
        .def(py::init([](py::object apply, bool is_complex, bool is_double) {
            return new HoppingModifier(
                [apply](ComplexArrayRef energy, CartesianArrayConstRef p1,
                        CartesianArrayConstRef p2, string_view hopping_family) {
                    py::gil_scoped_acquire guard{};
                    auto result = apply(
                        energy, arrayref(p1.x()), arrayref(p1.y()), arrayref(p1.z()),
                        arrayref(p2.x()), arrayref(p2.y()), arrayref(p2.z()), hopping_family
                    );
                    num::match<ArrayX>(energy, ExtractModifierResult{result});
                },
                is_complex, is_double
            );
        }), "apply"_a, "is_complex"_a=false, "is_double"_a=false)
        .def_readwrite("is_complex", &HoppingModifier::is_complex)
        .def_readwrite("is_double", &HoppingModifier::is_double);
}
