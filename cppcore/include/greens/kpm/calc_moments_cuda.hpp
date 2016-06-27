#pragma once
#include "greens/kpm/OptimizedSizes.hpp"

#include "detail/macros.hpp"
#include "numeric/sparseref.hpp"
#include "support/thrust.hpp"

namespace cpb { namespace cuda {

/**
 The Cuda functions must be defined only in nvcc-compiled translation units,
 but the declarations need to be visible to non-Cuda code as well. Since these
 are templates, they need to be explicitly instantiated in a Cuda translation
 unit for all relevant scalar types. To help with this, they are all wrapped in
 a template class `I`. This way, a single explicit instantiation of `I` will
 take care of everything. It's a bit weird but it works nicely.
*/
template<class scalar_t>
class I {
    using real_t = num::get_real_t<scalar_t>;
    using complex_t = num::get_complex_t<scalar_t>;
    
public:
    /**
     Diagonal KPM moments -- reference implementation, no optimizations

     Calculates moments for a single matrix element (i, i) on the main diagonal.
     It's 1.5x to 2x times faster than the general version.
     */
    static thr::host_vector<scalar_t>
        calc_diag_moments0(num::EllConstRef<scalar_t> ell, int i, int num_moments);

    /**
     Diagonal KPM moments -- with reordering optimization (optimal system size for each iteration)
     */
    static thr::host_vector<scalar_t>
        calc_diag_moments1(num::EllConstRef<scalar_t> ell, int i, int num_moments,
                           kpm::OptimizedSizes const& sizes);
};

CPB_EXTERN_TEMPLATE_CLASS(I)

}} // namespace cpb::cuda
