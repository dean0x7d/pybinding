#include "system/System.hpp"
#include "wrappers.hpp"
using namespace cpb;

template<class T>
void wrap_registry(py::module& m, char const* name) {
    py::class_<T>(m, name)
        .def_property_readonly("name_map", &T::name_map)
        .def(py::pickle([](T const& r) {
            return py::dict("energies"_a=r.get_energies(), "names"_a=r.get_names());
        }, [](py::dict d) {
            return new T(d["energies"].cast<std::vector<MatrixXcd>>(),
                         d["names"].cast<std::vector<std::string>>());
        }));
}

void wrap_system(py::module& m) {
    wrap_registry<SiteRegistry>(m, "SiteRegistry");
    wrap_registry<HoppingRegistry>(m, "HoppingRegistry");

    py::class_<CartesianArray>(m, "CartesianArray")
        .def_property_readonly("x", [](CartesianArray const& a) { return arrayref(a.x); })
        .def_property_readonly("y", [](CartesianArray const& a) { return arrayref(a.y); })
        .def_property_readonly("z", [](CartesianArray const& a) { return arrayref(a.z); })
        .def(py::pickle([](CartesianArray const& a) {
            return py::make_tuple(arrayref(a.x), arrayref(a.y), arrayref(a.z));
        }, [](py::tuple t) {
            using T = ArrayXf;
            return new CartesianArray(t[0].cast<T>(), t[1].cast<T>(), t[2].cast<T>());
        }));

    py::class_<CompressedSublattices>(m, "CompressedSublattices")
        .def("decompressed", [](CompressedSublattices const& c) { return c.decompressed(); })
        .def_property_readonly("alias_ids", &CompressedSublattices::alias_ids)
        .def_property_readonly("site_counts", &CompressedSublattices::site_counts)
        .def_property_readonly("orbital_counts", &CompressedSublattices::orbital_counts)
        .def(py::pickle([](CompressedSublattices const& c) {
            return py::dict("alias_ids"_a=c.alias_ids(), "site_counts"_a=c.site_counts(),
                            "orbital_counts"_a=c.orbital_counts());
        }, [](py::dict d) {
            return new CompressedSublattices(d["alias_ids"].cast<ArrayXi>(),
                                             d["site_counts"].cast<ArrayXi>(),
                                             d["orbital_counts"].cast<ArrayXi>());
        }));

    py::class_<HoppingBlocks>(m, "HoppingBlocks")
        .def_property_readonly("nnz", &HoppingBlocks::nnz)
        .def("count_neighbors", &HoppingBlocks::count_neighbors)
        .def("tocsr", [](HoppingBlocks const& hb) {
            auto type = py::module::import("pybinding.support.alias").attr("AliasCSRMatrix");
            return type(hb.tocsr(), "mapping"_a=hb.get_name_map());
        })
        .def("tocoo", [](py::object self) { return self.attr("tocsr")().attr("tocoo")(); })
        .def("__getitem__", [](py::object self, py::object item) {
            auto const structure = py::module::import("pybinding.support.structure");
            return structure.attr("Hoppings")(
                structure.attr("_slice_csr_matrix")(self.attr("tocsr")(), item)
            );
        })
        .def(py::pickle([](HoppingBlocks const& hb) {
            return py::dict("num_sites"_a=hb.get_num_sites(), "data"_a=hb.get_serialized_blocks(),
                            "name_map"_a=hb.get_name_map());
        }, [](py::dict d) {
            return new HoppingBlocks(d["num_sites"].cast<idx_t>(),
                                     d["data"].cast<HoppingBlocks::SerializedBlocks>(),
                                     d["name_map"].cast<NameMap>());
        }));

    using Boundary = System::Boundary;
    py::class_<Boundary>(m, "Boundary")
        .def_readonly("hoppings", &Boundary::hopping_blocks)
        .def_readonly("shift", &Boundary::shift)
        .def("__getitem__", [](py::object self, py::object item) {
            auto type = py::module::import("pybinding.support.structure").attr("Boundary");
            return type(self.attr("shift"), self.attr("hoppings")[item]);
        })
        .def(py::pickle([](Boundary const& b) {
            return py::make_tuple(b.hopping_blocks, b.shift);
        }, [](py::tuple t) {
            return new Boundary{t[0].cast<decltype(Boundary::hopping_blocks)>(),
                                t[1].cast<decltype(Boundary::shift)>()};
        }));

    py::class_<System, std::shared_ptr<System>>(m, "System")
        .def("find_nearest", &System::find_nearest, "position"_a, "sublattice"_a="")
        .def("to_hamiltonian_indices", &System::to_hamiltonian_indices)
        .def_readonly("site_registry", &System::site_registry)
        .def_readonly("hopping_registry", &System::hopping_registry)
        .def_readonly("positions", &System::positions)
        .def_readonly("compressed_sublattices", &System::compressed_sublattices)
        .def_readonly("hopping_blocks", &System::hopping_blocks)
        .def_readonly("boundaries", &System::boundaries)
        .def_property_readonly("hamiltonian_size", &System::hamiltonian_size)
        .def_property_readonly("expanded_positions", &System::expanded_positions)
        .def(py::pickle([](System const& s) {
            return py::dict("site_registry"_a=s.site_registry,
                            "hopping_registry"_a=s.hopping_registry,
                            "positions"_a=s.positions,
                            "compressed_sublattices"_a=s.compressed_sublattices,
                            "hopping_blocks"_a=s.hopping_blocks,
                            "boundaries"_a=s.boundaries);
        }, [](py::dict d) {
            auto s = [&]{
                if (d.contains("lattice")) {
                    auto const lattice = d["lattice"].cast<Lattice>();
                    return new System(lattice.site_registry(), lattice.hopping_registry());
                } else {
                    return new System(d["site_registry"].cast<SiteRegistry>(),
                                      d["hopping_registry"].cast<HoppingRegistry>());
                }
            }();
            s->positions = d["positions"].cast<decltype(s->positions)>();
            s->compressed_sublattices =
                d["compressed_sublattices"].cast<decltype(s->compressed_sublattices)>();
            s->hopping_blocks = d["hopping_blocks"].cast<decltype(s->hopping_blocks)>();
            s->boundaries = d["boundaries"].cast<decltype(s->boundaries)>();
            return s;
        }));
}
