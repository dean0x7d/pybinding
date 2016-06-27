#pragma once
#include "numeric/arrayref.hpp"

namespace cpb { namespace num {

namespace detail {
    /**
     Reference to CSR matrix of any type
     */
    struct BasicCsrConstRef {
        int const rows;
        int const cols;
        int const nnz;
        void const* const void_data;
        int const* const indices;
        int const* const indptr;
    };
} // namespace detail

/**
 Template reference to a CSR matrix with specific scalar type
 */
template<class scalar_t>
struct CsrConstRef : detail::BasicCsrConstRef {
    using type = scalar_t;

    CsrConstRef(int rows, int cols, int nnz, scalar_t const* data,
                int const* indices, int const* indptr)
        : detail::BasicCsrConstRef{rows, cols, nnz, data, indices, indptr} {}

    scalar_t const* data() const { return static_cast<scalar_t const*>(void_data); }
};

/**
 Tagged reference to a CSR matrix of any scalar type
 */
struct AnyCsrConstRef : detail::BasicCsrConstRef {
    Tag const tag;

    template<class scalar_t>
    AnyCsrConstRef(CsrConstRef<scalar_t> const& other)
        : detail::BasicCsrConstRef(other), tag(detail::get_tag<scalar_t>()) {}

    ArrayConstRef data_ref() const { return arrayref(tag, void_data, nnz); }
    ArrayConstRef indices_ref() const { return arrayref(indices, nnz); }
    ArrayConstRef indptr_ref() const { return arrayref(indptr, rows + 1); }
};

/**
 Template reference to a CSR matrix with a few possible scalar types
 */
template<class Scalar, class... Scalars>
struct VariantCsrConstRef : AnyCsrConstRef {
    using Types = TypeList<Scalar, Scalars...>;

    template<class scalar_t, class = std14::enable_if_t<tl::AnyOf<Types, scalar_t>::value>>
    VariantCsrConstRef(CsrConstRef<scalar_t> const& other) : AnyCsrConstRef(other) {}
};

/**
 Common VariantCsrConstRef aliases
 */
using RealCsrConstRef = VariantCsrConstRef<float, double>;
using ComplexCsrConstRef = VariantCsrConstRef<
    float, double, std::complex<float>, std::complex<double>
>;

namespace detail {
    /**
     Reference to ELLPACK matrix of any type
     */
    struct BasicEllConstRef {
        int const rows;
        int const cols;
        int const nnz_per_row;
        int const pitch;
        void const* const void_data;
        int const* const indices;

        int size() const { return nnz_per_row * pitch; }
    };
} // namespace detail

/**
 Template reference to an ELLPACK matrix with one specific scalar type
 */
template<class scalar_t>
struct EllConstRef : detail::BasicEllConstRef {
    using type = scalar_t;

    EllConstRef(int rows, int cols, int nnz_per_row, int pitch,
                scalar_t const* data, int const* indices)
        : detail::BasicEllConstRef{rows, cols, nnz_per_row, pitch, data, indices} {}

    scalar_t const* data() const { return static_cast<scalar_t const*>(void_data); }
};

/**
 Tagged reference to an ELLPACK matrix of any scalar type
 */
struct AnyEllConstRef : detail::BasicEllConstRef {
    Tag const tag;

    template<class scalar_t>
    AnyEllConstRef(EllConstRef<scalar_t> const& other)
        : detail::BasicEllConstRef(other), tag(num::detail::get_tag<scalar_t>()) {}

    ArrayConstRef data_ref() const { return arrayref(tag, void_data, size()); }
    ArrayConstRef indices_ref() const { return arrayref(indices, size()); }
};

/**
 Template reference to a ELLPACK matrix with a few possible scalar types
 */
template<class Scalar, class... Scalars>
struct VariantEllConstRef : AnyEllConstRef {
    using Types = TypeList<Scalar, Scalars...>;

    template<class scalar_t, class = std14::enable_if_t<tl::AnyOf<Types, scalar_t>::value>>
    VariantEllConstRef(EllConstRef<scalar_t> const& other) : AnyEllConstRef(other) {}
};

/**
 Common VariantEllConstRef aliases
 */
using RealEllConstRef = VariantEllConstRef<float, double>;
using ComplexEllConstRef = VariantEllConstRef<
    float, double, std::complex<float>, std::complex<double>
>;

}} // namespace cpb::num
