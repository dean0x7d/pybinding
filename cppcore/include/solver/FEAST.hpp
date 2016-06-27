#pragma once
#include "detail/config.hpp"

#ifdef CPB_USE_FEAST
#include "solver/Solver.hpp"

namespace cpb {

struct FEASTConfig {
    // required user config
    double energy_min = 0; ///< lowest eigenvalue
    double energy_max = 0; ///< highest eigenvalue
    int initial_size_guess = 0; ///< initial user guess for the subspace size

    // optional user config
    bool is_verbose = false; ///< [false] print information directly to stdout
    bool recycle_subspace = false; ///< [false] use previous data as a starting point

    // advanced optional user config
    int contour_points = 8; ///< [8] complex integral contour point
    int max_refinement_loops = 5; ///< [20] maximum number of refinement loops
    int sp_stop_criteria = 3; ///< [5] single precision error trace stopping criteria
    int dp_stop_criteria = 10; ///< [12] double precision error trace stopping criteria
    bool residual_convergence = false; /**< [false] use residual stop criteria
                                           instead of error trace criteria */

    // implementation detail config
    char matrix_format = 'F'; ///<  full matrix 'F' or triangular: lower 'L' and upper 'U'
    int system_size = 0; ///< size of the Hamiltonian matrix, i.e. number of atoms in the system
};

/**
 Implementation of the FEAST eigensolver
 */
template<class scalar_t>
class FEAST : public SolverStrategy {
    using real_t = num::get_real_t<scalar_t>;
    using complex_t = num::get_complex_t<scalar_t>;

public:
    struct Info {
        int suggested_size; ///< post-calculation suggested subspace size
        int final_size; ///< final subspace size
        int refinement_loops = 0; ///< the number of refinement loops executed
        real_t error_trace; ///< relative error on trace
        real_t max_residual; ///< biggest residual
        int return_code; ///< function return information and error codes
        bool recycle_warning = false; ///< error with recycled subspace, the calculation was rerun
        int recycle_warning_loops = 0;  ///< total loop count including those reset after a warning
        bool size_warning = false; ///< the initial subspace size was too small
    };
    
public:
    using Config = FEASTConfig;
    explicit FEAST(SparseMatrixRC<scalar_t> hamiltonian, Config const& config = {})
        : hamiltonian(std::move(hamiltonian)), config(config) {}

public: // overrides
    bool change_hamiltonian(Hamiltonian const& h) override;
    void solve() override;
    std::string report(bool shortform) const override;

    // map eigenvalues and wavefunctions to only expose results up to the usable subspace size
    RealArrayConstRef eigenvalues() const override {
        return arrayref(Map<const ArrayX<real_t>>(_eigenvalues.data(), info.final_size));
    }
    ComplexArrayConstRef eigenvectors() const override {
        return arrayref(Map<const ArrayXX<scalar_t>>(_eigenvectors.data(),
                                                     _eigenvectors.rows(), info.final_size));
    }

private: // implementation
    void init_feast(); ///< initialize FEAST parameters
    void init_pardiso(); ///< initialize PARDISO (sparse linear solver) parameters
    void call_feast(); ///< setup and call FEAST solver
    void call_feast_impl(); ///< call for scalar_t specific solver
    void force_clear(); ///< clear eigenvalue, eigenvector and residual data

private:
    SparseMatrixRC<scalar_t> hamiltonian;
    Config config;

    ArrayX<real_t> _eigenvalues;
    ArrayXX<scalar_t> _eigenvectors;

    int fpm[128]; ///< FEAST init parameters
    Info info;
    ArrayX<real_t> residual; ///< relative residual
};

extern template class cpb::FEAST<float>;
extern template class cpb::FEAST<std::complex<float>>;
extern template class cpb::FEAST<double>;
extern template class cpb::FEAST<std::complex<double>>;

} // namespace cpb
#endif // CPB_USE_FEAST
