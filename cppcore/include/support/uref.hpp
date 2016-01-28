#pragma once
#include "support/dense.hpp"
#include "support/sparse.hpp"

enum class ScalarType {f, cf, d, cd, b, i8, i16, i32, u8, u16, u32, none};

namespace detail {
    template<class scalar_t> constexpr ScalarType get_type();
    template<> constexpr ScalarType get_type<float>() { return ScalarType::f; }
    template<> constexpr ScalarType get_type<std::complex<float>>() { return ScalarType::cf; }
    template<> constexpr ScalarType get_type<double>() { return ScalarType::d; }
    template<> constexpr ScalarType get_type<std::complex<double>>() { return ScalarType::cd; }
    template<> constexpr ScalarType get_type<bool>() { return ScalarType::b; }
    template<> constexpr ScalarType get_type<std::int8_t>() { return ScalarType::i8; }
    template<> constexpr ScalarType get_type<std::int16_t>() { return ScalarType::i16; }
    template<> constexpr ScalarType get_type<std::int32_t>() { return ScalarType::i32; }
    template<> constexpr ScalarType get_type<std::uint8_t>() { return ScalarType::u8; }
    template<> constexpr ScalarType get_type<std::uint16_t>() { return ScalarType::u16; }
    template<> constexpr ScalarType get_type<std::uint32_t>() { return ScalarType::u32; }
}

struct DenseURef {
    ScalarType type; // this should be const but there's a bug on Intel 15.0
    const void* const data;
    const bool is_row_major;
    const int rows, cols;

    template<class Derived>
    DenseURef(const Eigen::DenseBase<Derived>& v)
        : type{::detail::get_type<typename Derived::Scalar>()},
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
    if (u.type != ::detail::get_type<typename Derived::Scalar>())
        throw std::logic_error{"eigen_cast(DenseURef) - wrong scalar type selected"};
    return ::detail::make_map<Derived>::exec(u);
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
