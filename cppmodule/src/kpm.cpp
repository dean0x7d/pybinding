#include "KPM.hpp"
#include "wrappers.hpp"
#include "thread.hpp"
using namespace cpb;

namespace {

void wrap_kpm_strategy(py::module& m, char const* name) {
    auto const kpm_defaults = kpm::Config();
    m.def(
        name,
        [](Model const& model, std::pair<float, float> energy, kpm::Kernel const& kernel,
           std::string matrix_format, bool optimal_size, bool interleaved, float lanczos,
           idx_t num_threads, kpm::DefaultCompute::ProgressCallback progress_callback) {
            kpm::Config config;
            config.min_energy = energy.first;
            config.max_energy = energy.second;
            config.kernel = kernel;
            config.matrix_format = matrix_format == "ELL" ? kpm::MatrixFormat::ELL
                                                          : kpm::MatrixFormat::CSR;
            config.algorithm.optimal_size = optimal_size;
            config.algorithm.interleaved = interleaved;
            config.lanczos_precision = lanczos;

            return KPM(model, kpm::DefaultCompute(num_threads, progress_callback), config);
        },
        "model"_a,
        "energy_range"_a=py::make_tuple(kpm_defaults.min_energy, kpm_defaults.max_energy),
        "kernel"_a=kpm_defaults.kernel,
        "matrix_format"_a="ELL",
        "optimal_size"_a=true,
        "interleaved"_a=true,
        "lanczos_precision"_a=kpm_defaults.lanczos_precision,
        "num_threads"_a=std::thread::hardware_concurrency(),
        "progress_callback"_a=py::none()
    );
}

struct ReturnMatrix {
    template<class T>
    ComplexCsrConstRef operator()(T) const { throw std::runtime_error("This will never happen"); }

    template<class scalar_t>
    ComplexCsrConstRef operator()(SparseMatrixX<scalar_t> const& h2) const { return csrref(h2); }
};

} // anonymously namespace

void wrap_greens(py::module& m) {
    py::class_<kpm::Stats>(m, "KPMStats")
        .def_readonly("num_moments", &kpm::Stats::num_moments)
        .def_readonly("uses_full_system", &kpm::Stats::uses_full_system)
        .def_readonly("nnz", &kpm::Stats::nnz)
        .def_readonly("opt_nnz", &kpm::Stats::opt_nnz)
        .def_readonly("vec", &kpm::Stats::vec)
        .def_readonly("opt_vec", &kpm::Stats::opt_vec)
        .def_readonly("matrix_memory", &kpm::Stats::matrix_memory)
        .def_readonly("vector_memory", &kpm::Stats::vector_memory)
        .def_property_readonly("eps", &kpm::Stats::eps)
        .def_property_readonly("ops", &kpm::Stats::ops)
        .def_property_readonly("hamiltonian_time", [](kpm::Stats const& s) {
            return s.hamiltonian_timer.elapsed_seconds();
        })
        .def_property_readonly("moments_time", [](kpm::Stats const& s) {
            return s.moments_timer.elapsed_seconds();
        });

    py::class_<kpm::Kernel>(m, "KPMKernel")
        .def_readonly("damping_coefficients", &kpm::Kernel::damping_coefficients)
        .def_readonly("required_num_moments", &kpm::Kernel::required_num_moments);

    m.def("lorentz_kernel", &kpm::lorentz_kernel);
    m.def("jackson_kernel", &kpm::jackson_kernel);
    m.def("dirichlet_kernel", &kpm::dirichlet_kernel);

    py::class_<KPM>(m, "KPM")
        .def("moments", &KPM::moments)
        .def("calc_greens", &KPM::calc_greens, release_gil())
        .def("calc_greens", &KPM::calc_greens_vector, release_gil())
        .def("calc_dos", &KPM::calc_dos, release_gil())
        .def("calc_conductivity", &KPM::calc_conductivity, release_gil())
        .def("calc_ldos", &KPM::calc_ldos, release_gil())
        .def("calc_spatial_ldos", &KPM::calc_spatial_ldos, release_gil())
        .def("deferred_ldos", [](py::object self, ArrayXd energy, double broadening,
                                 Cartesian position, std::string sublattice) {
            auto& kpm = self.cast<KPM&>();
            kpm.get_model().eval();
            return Deferred<ArrayXXdCM>{
                self, [=, &kpm] { return kpm.calc_ldos(energy, broadening, position, sublattice); }
            };
        })
        .def("report", &KPM::report, "shortform"_a=false)
        .def_property("model", &KPM::get_model, &KPM::set_model)
        .def_property_readonly("system", [](KPM const& kpm) { return kpm.get_model().system(); })
        .def_property_readonly("scaling_factors", [](KPM& kpm) {
            auto const s = kpm.get_core().scaling_factors();
            return py::make_tuple(s.a, s.b);
        })
        .def_property_readonly("kernel", [](KPM& kpm) {
            return kpm.get_core().get_config().kernel;
        })
        .def_property_readonly("stats", [](KPM& kpm) { return kpm.get_core().get_stats(); });

    wrap_kpm_strategy(m, "kpm");

    py::class_<kpm::OptimizedHamiltonian>(m, "OptimizedHamiltonian")
        .def("__init__", [](kpm::OptimizedHamiltonian& self, Hamiltonian const& h, int index) {
            new (&self) kpm::OptimizedHamiltonian(h, kpm::MatrixFormat::CSR, true);

            auto indices = std::vector<idx_t>(h.rows());
            std::iota(indices.begin(), indices.end(), 0);
            auto bounds = kpm::Bounds(h, kpm::Config{}.lanczos_precision);
            self.optimize_for({index, indices}, bounds.scaling_factors());
        })
        .def_property_readonly("matrix", [](kpm::OptimizedHamiltonian const& self) {
            return var::apply_visitor(ReturnMatrix{}, self.matrix());
        })
        .def_property_readonly("sizes", [](kpm::OptimizedHamiltonian const& self) {
            return self.map().get_data();
        })
        .def_property_readonly("indices", [](kpm::OptimizedHamiltonian const& self) {
            return self.idx().dest;
        });
}
