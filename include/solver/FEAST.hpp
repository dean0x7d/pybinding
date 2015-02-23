#pragma once
#include "support/config.hpp"
#include "solver/Solver.hpp"

#ifdef TBM_USE_FEAST
namespace tbm {

/**
 Implementation of the FEAST eigensolver.
 */
template<typename scalar_t>
class FEAST : public SolverT<scalar_t> {
    using real_t = num::get_real_t<scalar_t>;
    using complex_t = num::get_complex_t<scalar_t>;
public:
    
    struct Params {
        real_t energy_min, energy_max; ///< solution energy range
        int initial_size_guess; ///< initial user guess for the subspace size
        char matrix_format = 'F'; ///<  full matrix 'F' or triangular: lower 'L' and upper 'U'
        int system_size; ///< size of the Hamiltonian matrix, i.e. number of atoms in the system
        
        bool is_verbose = false; ///< [false] print information directly to stdout
        bool recycled_subspace = false; ///< [false] use previous data as a starting point
        int contour_points = 8; ///< [8] complex integral contour point
        int max_refinement_loops = 5; ///< [20] maximum number of refinement loops
        int sp_stop_criteria = 3; ///< [5] single precision error trace stopping criteria
        int dp_stop_criteria = 10; ///< [12] double precision error trace stopping criteria
        bool residual_convergence = false; /**< [false] use residual stop criteria
                                           instead of error trace criteria */
    };
    
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
    
protected:
    FEAST(real_t energy_min, real_t energy_max, int subspace_size_guess,
          bool recycle_subspace, bool is_verbose);

public: // overrides
    // map eigenvalues and wavefunctions to only expose results up to the usable subspace size
    virtual DenseURef eigenvalues() const override {
        return Map<const ArrayX<real_t>>(_eigenvalues.data(), info.final_size);
    }
    virtual DenseURef eigenvectors() const override {
        return Map<const ArrayXX<scalar_t>>(_eigenvectors.data(), _eigenvectors.rows(), info.final_size);
    }

    virtual std::string v_report(bool shortform) const override;
    /// FEAST will not clear data if recycle_subspace == true
    virtual void clear() override;

protected: // overrides
    virtual void v_solve() override;

private: // implementation
    void init_feast(); ///< initialize FEAST parameters
    void init_pardiso(); ///< initialize PARDISO (sparse linear solver) parameters
    void call_feast(); ///< setup and call FEAST solver
    void call_feast_impl(); ///< call for scalar_t specific solver
    void force_clear(); ///< clear eigenvalue, eigenvector and residual data

private:
    int	fpm[128]; ///< FEAST init parameters
    Params params;
    Info info;
    ArrayX<real_t> residual; ///< relative residual

protected: // declared used inherited members (template class requirement)
    using Solver::is_solved;
    using SolverT<scalar_t>::_eigenvalues;
    using SolverT<scalar_t>::_eigenvectors;
    using SolverT<scalar_t>::hamiltonian;

    friend class FEASTFactory;
};


/**
 Concrete SolverFactory for creating FEAST objects.
 */
class FEASTFactory : public SolverFactory {
public:
    struct defaults {
        static constexpr bool recycle_subspace = false;
        static constexpr bool is_verbose = false;
    };
    
    /**
     Find the eigenvalues and eigenvectors in the given energy range.
     
     @param energy_min Bottom limit of the energy range.
     @param energy_max Top limit of the energy range.
     @param subspace_size_guess Initial guess for the number of states in the energy range.
     The optimal value should be 50% bigger than final subspace size.
     @param recycle_subspace Reuse previous results as initial data for the solver.
     @param is_verbose Activate FEAST solver info (prints directly to stdout).
     */
    FEASTFactory(double energy_min, double energy_max, int subspace_size_guess,
                 bool recycle_subspace = defaults::recycle_subspace,
                 bool is_verbose = defaults::is_verbose)
        : energy_min{energy_min}, energy_max{energy_max}, subspace_size{subspace_size_guess},
          recycle_subspace{recycle_subspace}, is_verbose{is_verbose}
    {}
    
    // required implementation
    virtual std::unique_ptr<Solver> 
        create_for(const std::shared_ptr<const Hamiltonian>& hamiltonian) const override;
    /// Set the advanced options of the FEAST solver
    FEASTFactory& advanced(int points, int loops, int sp, int dp, bool stop_residual);
    
private: // create template
    template<typename scalar_t>
    std::unique_ptr<Solver> try_create_for(const std::shared_ptr<const Hamiltonian>& h) const;
    
private:
    // basic
    double energy_min, energy_max;
    int subspace_size;
    bool recycle_subspace, is_verbose;
    
    // advanced
    FEAST<double>::Params params;
};

} // namespace tbm
#endif // TBM_USE_FEAST
