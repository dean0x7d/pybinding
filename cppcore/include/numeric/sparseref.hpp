#pragma once
#include "numeric/arrayref.hpp"

namespace tbm { namespace num {

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

/**
 Template reference to CSR matrix specific type
 */
template<class scalar_t = void>
struct CsrConstRef : BasicCsrConstRef {
    using type = scalar_t;

    CsrConstRef(int rows, int cols, int nnz, scalar_t const* data,
                int const* indices, int const* indptr)
        : BasicCsrConstRef{rows, cols, nnz, data, indices, indptr} {}

    scalar_t const* data() const { return static_cast<scalar_t const*>(void_data); }
};

/**
 Tagged reference to CSR matrix specific type
 */
template<>
struct CsrConstRef<void> : BasicCsrConstRef {
    Tag const tag;

    template<class scalar_t>
    CsrConstRef(CsrConstRef<scalar_t> const& other)
        : BasicCsrConstRef(other), tag(detail::get_tag<scalar_t>()) {}

    ArrayConstRef data_ref() const { return arrayref(tag, void_data, nnz); }
    ArrayConstRef indices_ref() const { return arrayref(indices, nnz); }
    ArrayConstRef indptr_ref() const { return arrayref(indptr, rows + 1); }
};

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

/**
 Template reference to ELLPACK matrix specific type
 */
template<class scalar_t = void>
struct EllConstRef : BasicEllConstRef {
    using type = scalar_t;

    EllConstRef(int rows, int cols, int nnz_per_row, int pitch,
                scalar_t const* data, int const* indices)
        : BasicEllConstRef{rows, cols, nnz_per_row, pitch, data, indices} {}

    scalar_t const* data() const { return static_cast<scalar_t const*>(void_data); }
};

/**
 Tagged reference to ELLPACK matrix specific type
 */
template<>
struct EllConstRef<void> : BasicEllConstRef {
    Tag const tag;

    template<class scalar_t>
    EllConstRef(EllConstRef<scalar_t> const& other)
        : BasicEllConstRef(other), tag(num::detail::get_tag<scalar_t>()) {}

    ArrayConstRef data_ref() const { return arrayref(tag, void_data, size()); }
    ArrayConstRef indices_ref() const { return arrayref(indices, size()); }
};

}} // namespace tbm::num
