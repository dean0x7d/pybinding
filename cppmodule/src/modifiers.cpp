#include "system/SystemModifiers.hpp"
#include "system/Generators.hpp"
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

template<class EigenType>
void extract_modifier_result(EigenType& v, py::object const& o) {
    ExtractModifierResult{o}(Eigen::Map<EigenType>(v.data(), v.size()));
}

} // anonymous namespace

void wrap_modifiers(py::module& m) {
    py::class_<SubIdRef>(m, "SubIdRef")
        .def_property_readonly("ids", [](SubIdRef const& s) { return arrayref(s.ids); })
        .def_readonly("name_map", &SubIdRef::name_map);

    py::class_<HopIdRef>(m, "HopIdRef")
        .def_property_readonly("ids", [](HopIdRef const& s) { return arrayref(s.ids); })
        .def_readonly("name_map", &HopIdRef::name_map);

    py::class_<SiteStateModifier>(m, "SiteStateModifier")
        .def("__init__", [](SiteStateModifier& self, py::object apply, int min_neighbors) {
            new (&self) SiteStateModifier(
                [apply](ArrayX<bool>& state, CartesianArray const& p, SubIdRef sub) {
                    auto result = apply(arrayref(state), arrayref(p.x),
                                        arrayref(p.y), arrayref(p.z), sub);
                    extract_modifier_result(state, result);
                },
                min_neighbors
            );
        }, "apply"_a, "min_neighbors"_a=0);

    py::class_<PositionModifier>(m, "PositionModifier")
        .def("__init__", [](PositionModifier& self, py::object apply) {
            new (&self) PositionModifier([apply](CartesianArray& p, SubIdRef sub) {
                auto t = py::tuple(apply(arrayref(p.x), arrayref(p.y), arrayref(p.z), sub));
                extract_modifier_result(p.x, t[0]);
                extract_modifier_result(p.y, t[1]);
                extract_modifier_result(p.z, t[2]);
            });
        });

    py::class_<HoppingGenerator>(m, "HoppingGenerator")
        .def("__init__", [](HoppingGenerator& self, std::string const& name,
                            std::complex<double> energy, py::object make) {
            new (&self) HoppingGenerator(
                name, energy,
                [make](CartesianArray const& p, SubIdRef sub) {
                    auto t = py::tuple(make(arrayref(p.x), arrayref(p.y), arrayref(p.z), sub));
                    return HoppingGenerator::Result{t[0].cast<ArrayXi>(), t[1].cast<ArrayXi>()};
                }
            );
        });

    py::class_<OnsiteModifier>(m, "OnsiteModifier")
        .def("__init__", [](OnsiteModifier& self, py::object apply,
                            bool is_complex, bool is_double) {
            new (&self) OnsiteModifier(
                [apply](ComplexArrayRef energy, CartesianArrayConstRef p, SubIdRef sub) {
                    auto result = apply(energy, arrayref(p.x()), arrayref(p.y()),
                                        arrayref(p.z()), sub);
                    num::match<ArrayX>(energy, ExtractModifierResult{result});
                },
                is_complex, is_double
            );
        }, "apply"_a, "is_complex"_a=false, "is_double"_a=false)
        .def_readwrite("is_complex", &OnsiteModifier::is_complex)
        .def_readwrite("is_double", &OnsiteModifier::is_double);

    py::class_<HoppingModifier>(m, "HoppingModifier")
        .def("__init__", [](HoppingModifier& self, py::object apply,
                            bool is_complex, bool is_double) {
            new (&self) HoppingModifier(
                [apply](ComplexArrayRef energy, CartesianArrayConstRef p1,
                        CartesianArrayConstRef p2, HopIdRef hopping) {
                    auto result = apply(
                        energy, arrayref(p1.x()), arrayref(p1.y()), arrayref(p1.z()),
                        arrayref(p2.x()), arrayref(p2.y()), arrayref(p2.z()), hopping
                    );
                    num::match<ArrayX>(energy, ExtractModifierResult{result});
                },
                is_complex, is_double
            );
        }, "apply"_a, "is_complex"_a=false, "is_double"_a=false)
        .def_readwrite("is_complex", &HoppingModifier::is_complex)
        .def_readwrite("is_double", &HoppingModifier::is_double);
}
