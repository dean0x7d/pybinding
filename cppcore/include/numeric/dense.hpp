#pragma once
#include "detail/config.hpp"
#include "numeric/traits.hpp"
#include "numeric/arrayref.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <numeric>
#include <vector>

namespace cpb {

// add common math functions to the global namespace
using std::abs;
using std::exp;
using std::pow;
using std::sqrt;
using std::sin;
using std::cos;
using std::tan;
using std::asin;
using std::acos;

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
using Eigen::Vector3f;
using Eigen::VectorXi;
using Eigen::VectorXf;
using Eigen::VectorXcf;
using Eigen::VectorXd;
using Eigen::VectorXcd;
using Eigen::MatrixXf;
using Eigen::MatrixXcf;
using Eigen::MatrixXd;
using Eigen::MatrixXcd;

// convenient type aliases
using Cartesian = Eigen::Vector3f;
using Index3D = Eigen::Vector3i;
using Vector3b = Eigen::Matrix<bool, 3, 1>;
template<class T> using ArrayX = Eigen::Array<T, Eigen::Dynamic, 1>;
template<class T> using ArrayXX = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
template<class T> using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template<class T> using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template<class T>
using ColMajorArrayXX = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
template<class T>
using ColMajorMatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

using ArrayXXdCM = ColMajorArrayXX<double>;

// array variants
using num::arrayref;
using num::ArrayConstRef;
using num::RealArrayConstRef;
using num::ComplexArrayConstRef;
using num::ArrayRef;
using num::RealArrayRef;
using num::ComplexArrayRef;

} // namespace cpb

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

namespace cpb {

/**
 Map std::vector-like object data to an Eigen type
 */
template<template<class> class EigenType, class Vector,
         class scalar_t = typename Vector::value_type>
inline Eigen::Map<EigenType<scalar_t> const> eigen_cast(Vector const& v) {
    return {v.data(), static_cast<idx_t>(v.size())};
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

/// Apply the function to the elements of the input container
/// and return the results in a new container of the given type
template<template<class...> class Container, class In, class Fn>
auto transform(In const& in, Fn func) -> Container<decltype(func(in[0]))> {
    using Out = decltype(func(in[0]));
    auto out = Container<Out>(in.size());
    std::transform(begin(in), end(in), begin(out), func);
    return out;
}

template<class Derived> inline bool any_of(const DenseBase<Derived>& v) { return v.any(); }
template<class Derived> inline bool all_of(const DenseBase<Derived>& v) { return v.all(); }
template<class Derived> inline bool none_of(const DenseBase<Derived>& v) { return !v.any(); }

template<bool is_const>
class BasicCartesianArrayRef {
public:
    using Reference = std14::conditional_t<!is_const, Eigen::Ref<ArrayXf>,
                                           Eigen::Ref<ArrayXf const>>;

    BasicCartesianArrayRef(Reference x, Reference y,  Reference z)
        : x_ref(x), y_ref(y), z_ref(z) {}

    BasicCartesianArrayRef(BasicCartesianArrayRef<false> const& o)
        : x_ref(o.x()), y_ref(o.y()), z_ref(o.z()) {}

    Reference const& x() const { return x_ref; }
    Reference const& y() const { return y_ref; }
    Reference const& z() const { return z_ref; }
    Reference& x() { return x_ref; }
    Reference& y() { return y_ref; }
    Reference& z() { return z_ref; }

    idx_t size() const { return x_ref.size(); }

private:
    Reference x_ref, y_ref, z_ref;
};

using CartesianArrayConstRef = BasicCartesianArrayRef<true>;
using CartesianArrayRef = BasicCartesianArrayRef<false>;

class CartesianArray {
private:
    struct CartesianRef {
        float &x, &y, &z;
        CartesianRef& operator=(const Cartesian& r) { x = r[0]; y = r[1]; z = r[2]; return *this; }
        operator Cartesian() { return {x, y, z}; }
    };

public:
    CartesianArray() = default;
    CartesianArray(idx_t size) : x(size), y(size), z(size) {}
    CartesianArray(ArrayXf x, ArrayXf y, ArrayXf z)
        : x(std::move(x)), y(std::move(y)), z(std::move(z)) {}

    CartesianRef operator[](idx_t i) { return {x[i], y[i], z[i]}; }
    Cartesian operator[](idx_t i) const { return {x[i], y[i], z[i]}; }

    idx_t size() const { return x.size(); }

    CartesianArrayConstRef head(idx_t size) const {
        return {x.head(size), y.head(size), z.head(size)};
    }

    CartesianArrayRef head(idx_t size) {
        return {x.head(size), y.head(size), z.head(size)};
    }

    CartesianArrayConstRef segment(idx_t start, idx_t size) const {
        return {x.segment(start, size), y.segment(start, size), z.segment(start, size)};
    }

    CartesianArrayRef segment(idx_t start, idx_t size) {
        return {x.segment(start, size), y.segment(start, size), z.segment(start, size)};
    }

    operator CartesianArrayConstRef() const { return {x, y, z}; }
    operator CartesianArrayRef() { return {x, y, z}; }

    template<class Fn>
    void for_each(Fn lambda) {
        lambda(x); lambda(y); lambda(z);
    }

    void resize(idx_t size) {
        for_each([size](ArrayX<float>& a) { a.resize(size); });
    }

    void conservativeResize(idx_t size) {
        for_each([size](ArrayX<float>& a) { a.conservativeResize(size); });
    }

public:
    ArrayX<float> x, y, z;
};

namespace num {
    // ArrayRef's MakeContainer specializations for Eigen types
    template<template<class, int...> class EigenType, class scalar_t, int cols, int... options>
    struct MakeContainer<EigenType<scalar_t, 1, cols, options...>> {
        using ConstMap = Eigen::Map<const EigenType<scalar_t, 1, cols, options...>>;
        static ConstMap make(ArrayConstRef const& ref) {
            return ConstMap{static_cast<scalar_t const*>(ref.data), ref.size()};
        }
        using Map = Eigen::Map<EigenType<scalar_t, 1, cols, options...>>;
        static Map make(ArrayRef const& ref) {
            return Map{static_cast<scalar_t*>(ref.data), ref.size()};
        }
    };

    template<template<class, int...> class EigenType, class scalar_t, int rows, int... options>
    struct MakeContainer<EigenType<scalar_t, rows, 1, options...>> {
        using ConstMap = Eigen::Map<const EigenType<scalar_t, rows, 1, options...>>;
        static ConstMap make(ArrayConstRef const& ref) {
            return ConstMap{static_cast<scalar_t const*>(ref.data), ref.size()};
        }
        using Map = Eigen::Map<EigenType<scalar_t, rows, 1, options...>>;
        static Map make(ArrayRef const& ref) {
            return Map{static_cast<scalar_t*>(ref.data), ref.size()};
        }
    };

    template<template<class, int...> class EigenType,
             class scalar_t, int rows, int cols, int... options>
    struct MakeContainer<EigenType<scalar_t, rows, cols, options...>> {
        using ConstMap = Eigen::Map<const EigenType<scalar_t, rows, cols, options...>>;
        static ConstMap make(ArrayConstRef const& ref) {
            return ConstMap{static_cast<scalar_t const*>(ref.data), ref.shape[0], ref.shape[1]};
        }
        using Map = Eigen::Map<EigenType<scalar_t, rows, cols, options...>>;
        static Map make(ArrayRef const& ref) {
            return Map{static_cast<scalar_t*>(ref.data), ref.shape[0], ref.shape[1]};
        }
    };

    /// Force cast a matrix to any scalar type (lose precision and/or imaginary part)
    template<class scalar_t>
    MatrixX<scalar_t> force_cast(MatrixXcd const& m) { return m.cast<scalar_t>(); }
    template<>
    inline MatrixX<double> force_cast<double>(MatrixXcd const& m) { return m.real(); }
    template<>
    inline MatrixX<float> force_cast<float>(MatrixXcd const& m) { return m.real().cast<float>(); }

    template<class scalar_t>
    VectorX<scalar_t> force_cast(VectorXcd const& m) { return m.cast<scalar_t>(); }
    template<>
    inline VectorX<double> force_cast<double>(VectorXcd const& m) { return m.real(); }
    template<>
    inline VectorX<float> force_cast<float>(VectorXcd const& m) { return m.real().cast<float>(); }
} // namespace num

template<class Derived>
ArrayConstRef arrayref(DenseBase<Derived> const& v) {
    auto const& d = v.derived();
    return {v.derived().data(),
            Derived::IsVectorAtCompileTime ? 1 : 2,
            Derived::IsRowMajor,
            Derived::IsVectorAtCompileTime ? d.size() : d.rows(),
            Derived::IsVectorAtCompileTime ? 0        : d.cols()};
};

template<class Derived>
auto arrayref(DenseBase<Derived>& v) -> num::detail::MakeArrayRef<decltype(v.derived().data())> {
    auto& d = v.derived();
    return {v.derived().data(),
            Derived::IsVectorAtCompileTime ? 1 : 2,
            Derived::IsRowMajor,
            Derived::IsVectorAtCompileTime ? d.size() : d.rows(),
            Derived::IsVectorAtCompileTime ? 0        : d.cols()};
};

template<class scalar_t>
ArrayConstRef arrayref(std::vector<scalar_t> const& v) {
    return arrayref(v.data(), static_cast<idx_t>(v.size()));
}

template<class scalar_t>
ArrayRef arrayref(std::vector<scalar_t>& v) {
    return arrayref(v.data(), 1, true, static_cast<idx_t>(v.size()));
}

/// Range from 0 to `size` of scalar type `T` which does not have to be an integral type
template<class T>
ArrayX<T> make_integer_range(idx_t size) {
    auto result = ArrayX<T>(size);
    for (auto n = 0; n < size; ++n) {
        result[n] = static_cast<T>(n);
    }
    return result;
}

template<class Vector, class Bools>
Vector slice(Vector const& v, Bools const& keep) {
    using std::begin; using std::end;
    auto const original_size = v.size();
    auto const result_size = std::accumulate(begin(keep), end(keep), idx_t{0});

    auto result = Vector(result_size);
    auto count = 0;
    for (auto i = idx_t{0}; i < original_size; ++i) {
        if (keep[i])
            result[count++] = v[i];
    }
    return result;
};

/// Concatenate two 1D arrays/vectors
template<class Vector>
Vector concat(Ref<Vector const> v1, Ref<Vector const> v2) {
    using std::begin; using std::end;
    auto result = Vector(v1.size() + v2.size());
    std::copy(begin(v1), end(v1), begin(result));
    std::copy(begin(v2), end(v2), begin(result) + v1.size());
    return result;
}

template<class Vector, typename R = Ref<Vector const>>
Vector concat(Vector const& v1, Vector const& v2) {
    return concat(R(v1), R(v2));
}

/// Concatenate two Cartesian arrays
inline CartesianArray concat(CartesianArrayConstRef const& ca1,
                             CartesianArrayConstRef const& ca2) {
    return {concat(ca1.x(), ca2.x()),
            concat(ca1.y(), ca2.y()),
            concat(ca1.z(), ca2.z())};
}

} // namespace cpb
