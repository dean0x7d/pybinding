#pragma once
#include "support/config.hpp"
#include "support/traits.hpp"
#include "support/arrayref.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <vector>

namespace tbm {

// add common math functions to the global namespace
using std::abs;
using std::exp;
using std::pow;
using std::sqrt;

// add common Eigen types to the global namespace
using Eigen::Ref;
using Eigen::Map;
using Eigen::DenseBase;
using Eigen::Array3i;
using Eigen::ArrayXi;
using Eigen::ArrayXf;
using Eigen::ArrayXcf;
using Eigen::ArrayXd;
using Eigen::ArrayXcd;
using Eigen::ArrayXXi;
using Eigen::ArrayXXf;
using Eigen::ArrayXXcf;
using Eigen::ArrayXXd;
using Eigen::ArrayXXcd;
using Eigen::VectorXi;
using Eigen::VectorXf;
using Eigen::VectorXcf;
using Eigen::VectorXd;
using Eigen::VectorXcd;

// convenient type aliases
using Cartesian = Eigen::Vector3f;
using Index3D = Eigen::Vector3i;
template<class T> using ArrayX = Eigen::Array<T, Eigen::Dynamic, 1>;
template<class T> using ArrayXX = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
template<class T> using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template<class T> using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

// array variants
using num::ArrayRef;
using num::RealArrayRef;
using num::ComplexArrayRef;

} // namespace tbm

namespace Eigen {
    // add being() and end() to Eigen namespace
    // this will enable using Eigen objects in ranged for loops
    template<class Derived>
    inline auto begin(EigenBase<Derived>& v) -> decltype(v.derived().data()) {
        return v.derived().data();
    }

    template<class Derived>
    inline auto end(EigenBase<Derived>& v) -> decltype(v.derived().data()) {
        return v.derived().data() + v.size();
    }

    template<class Derived>
    inline auto begin(const EigenBase<Derived>& v) -> decltype(v.derived().data()) {
        return v.derived().data();
    }

    template<class Derived>
    inline auto end(const EigenBase<Derived>& v) -> decltype(v.derived().data()) {
        return v.derived().data() + v.size();
    }
} // namespace Eigen

namespace tbm {

template<template<class> class EigenType, class scalar_t>
Eigen::Map<const EigenType<scalar_t>> eigen_cast(const std::vector<scalar_t>& v) {
    return Eigen::Map<const EigenType<scalar_t>>(v.data(), v.size());
}

// utility functions
template<class Derived>
inline auto sum(const DenseBase<Derived>& v) -> decltype(v.sum()) {
    return v.sum();
}

template<class DerivedIn, class DerivedOut, class Fn>
inline void transform(const DenseBase<DerivedIn>& in, DenseBase<DerivedOut>& out, Fn func) {
    std::transform(begin(in), end(in), begin(out), func);
}

template<class DerivedIn1, class DerivedIn2, class DerivedOut, class Fn>
inline void transform(const DenseBase<DerivedIn1>& in1, const DenseBase<DerivedIn2>& in2,
                      DenseBase<DerivedOut>& out, Fn func) {
    std::transform(begin(in1), end(in1), begin(in2), begin(out), func);
}

template<class Derived> inline bool any_of(const DenseBase<Derived>& v) { return v.any(); }
template<class Derived> inline bool all_of(const DenseBase<Derived>& v) { return v.all(); }
template<class Derived> inline bool none_of(const DenseBase<Derived>& v) { return !v.any(); }


class CartesianArray {
private:
    struct CartesianRef {
        float &x, &y, &z;
        CartesianRef& operator=(const Cartesian& r) { x = r[0]; y = r[1]; z = r[2]; return *this; }
        operator Cartesian() { return {x, y, z}; }
    };

public:
    CartesianArray() = default;
    CartesianArray(int size) : x(size), y(size), z(size) {}
    CartesianArray(ArrayXf const& x, ArrayXf const& y, ArrayXf const& z) : x(x), y(y), z(z) {}

    CartesianRef operator[](int i) { return {x[i], y[i], z[i]}; }
    Cartesian operator[](int i) const { return {x[i], y[i], z[i]}; }

    int size() const { return static_cast<int>(x.size()); }

    template<class Fn>
    void for_each(Fn lambda) {
        lambda(x); lambda(y); lambda(z);
    }

    void resize(int size) {
        for_each([size](ArrayX<float>& a) { a.resize(size); });
    }

    void conservativeResize(int size) {
        for_each([size](ArrayX<float>& a) { a.conservativeResize(size); });
    }

public:
    ArrayX<float> x, y, z;
};

namespace num {
    // ArrayRef's MakeContainer specializations for Eigen types
    template<template<class, int...> class EigenType, class scalar_t, int cols, int... options>
    struct MakeContainer<EigenType<scalar_t, 1, cols, options...>> {
        using Map = Eigen::Map<const EigenType<scalar_t, 1, cols, options...>>;

        static Map make(ArrayRef const& ref) {
            return Map{static_cast<scalar_t const*>(ref.data), ref.cols};
        }
    };

    template<template<class, int...> class EigenType, class scalar_t, int rows, int... options>
    struct MakeContainer<EigenType<scalar_t, rows, 1, options...>> {
        using Map = Eigen::Map<const EigenType<scalar_t, rows, 1, options...>>;

        static Map make(ArrayRef const& ref) {
            return Map{static_cast<scalar_t const*>(ref.data), ref.rows};
        }
    };

    template<template<class, int...> class EigenType,
             class scalar_t, int rows, int cols, int... options>
    struct MakeContainer<EigenType<scalar_t, rows, cols, options...>> {
        using Map = Eigen::Map<const EigenType<scalar_t, rows, cols, options...>>;

        static Map make(ArrayRef const& ref) {
            return Map{static_cast<scalar_t const*>(ref.data), ref.rows, ref.cols};
        }
    };
} // namespace num

template<class Derived>
inline ArrayRef arrayref(DenseBase<Derived> const& v) {
    return {num::detail::get_tag<typename Derived::Scalar>(),
            Derived::IsRowMajor,
            v.derived().data(),
            static_cast<int>(v.derived().rows()),
            static_cast<int>(v.derived().cols())};
};

template<class scalar_t>
inline ArrayRef arrayref(scalar_t const* data, int size) {
    return {num::detail::get_tag<scalar_t>(), true, data, 1, size};
};

} // namespace tbm
