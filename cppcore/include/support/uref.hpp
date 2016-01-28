#pragma once
#include "support/dense.hpp"
#include "support/sparse.hpp"

struct DenseURef {
    tbm::num::Tag type; // this should be const but there's a bug on Intel 15.0
    const void* const data;
    const bool is_row_major;
    const int rows, cols;

    template<class Derived>
    DenseURef(const Eigen::DenseBase<Derived>& v)
        : type{tbm::num::detail::get_tag<typename Derived::Scalar>()},
          data(v.derived().data()), is_row_major(Derived::IsRowMajor),
          rows(static_cast<int>(v.derived().rows())),
          cols(static_cast<int>(v.derived().cols()))
    {}
};

namespace detail {
    template<class Derived, bool is_vector = Derived::IsVectorAtCompileTime> struct make_map;

    template<class Derived>
    struct make_map<Derived, true> {
        static Eigen::Map<const Derived> exec(const DenseURef& u) {
            using data_t = const typename Derived::Scalar;
            return {static_cast<data_t*>(u.data), u.is_row_major ? u.cols : u.rows};
        }
    };

    template<class Derived>
    struct make_map<Derived, false> {
        static Eigen::Map<const Derived> exec(const DenseURef& u) {
            using data_t = const typename Derived::Scalar;
            return {static_cast<data_t*>(u.data), u.rows, u.cols};
        }
    };
}

template<class Derived>
Eigen::Map<const Derived> uref_cast(const DenseURef& u) {
    if (u.type != tbm::num::detail::get_tag<typename Derived::Scalar>())
        throw std::logic_error{"eigen_cast(DenseURef) - wrong scalar type selected"};
    return ::detail::make_map<Derived>::exec(u);
}

inline tbm::num::BasicArrayRef make_arrayvariant(DenseURef const& u) {
    return {u.type, u.is_row_major, u.data, u.rows, u.cols};
}

struct SparseURef {
    const DenseURef values;
    const DenseURef inner_indices;
    const DenseURef outer_starts;
    const int rows, cols;

    template<class scalar_t>
    SparseURef(const tbm::SparseMatrixX<scalar_t>& v)
        : values{Eigen::Map<const tbm::ArrayX<scalar_t>>(v.valuePtr(), v.nonZeros())},
          inner_indices{Eigen::Map<const tbm::ArrayXi>(v.innerIndexPtr(), v.nonZeros())},
          outer_starts{Eigen::Map<const tbm::ArrayXi>(v.outerIndexPtr(), v.outerSize() + 1)},
          rows(v.rows()), cols(v.cols())
    {}
};
