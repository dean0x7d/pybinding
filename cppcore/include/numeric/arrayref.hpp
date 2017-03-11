#pragma once
#include "detail/typelist.hpp"
#include "support/cppfuture.hpp"
#include "numeric/traits.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <array>

namespace cpb { namespace num {

/// Array scalar type
enum class Tag : std::int8_t {f32, cf32, f64, cf64, b, i8, i16, i32, i64, u8, u16, u32, u64};

namespace detail {
    template<class scalar_t> constexpr Tag get_tag();
    template<> constexpr Tag get_tag<float>()                { return Tag::f32;  }
    template<> constexpr Tag get_tag<std::complex<float>>()  { return Tag::cf32; }
    template<> constexpr Tag get_tag<double>()               { return Tag::f64;  }
    template<> constexpr Tag get_tag<std::complex<double>>() { return Tag::cf64; }
    template<> constexpr Tag get_tag<bool>()                 { return Tag::b;    }
    template<> constexpr Tag get_tag<std::int8_t>()          { return Tag::i8;   }
    template<> constexpr Tag get_tag<std::int16_t>()         { return Tag::i16;  }
    template<> constexpr Tag get_tag<std::int32_t>()         { return Tag::i32;  }
    template<> constexpr Tag get_tag<std::int64_t>()         { return Tag::i64;  }
    template<> constexpr Tag get_tag<std::uint8_t>()         { return Tag::u8;   }
    template<> constexpr Tag get_tag<std::uint16_t>()        { return Tag::u16;  }
    template<> constexpr Tag get_tag<std::uint32_t>()        { return Tag::u32;  }
    template<> constexpr Tag get_tag<std::uint64_t>()        { return Tag::u64;  }

    template<class T>
    constexpr Tag get_tag() {
        static_assert(std::is_integral<T>::value && (sizeof(T) == 8 || sizeof(T) == 4), "");
        using type = std14::conditional_t<
            std::is_signed<T>::value,
            std14::conditional_t<sizeof(T) == 8, std::int64_t, std::int32_t>,
            std14::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>
        >;
        return get_tag<type>();
    }

    /// Reference to any 1D, 2D or 3D array with any scalar type supported by Tag
    template<bool is_const>
    struct BasicArrayRef {
        using Ptr = std14::conditional_t<is_const, void const*, void*>;
        using Dim = std::int8_t;
        using Shape = std::array<idx_t, 3>;

        Ptr data;
        Tag tag;
        Dim ndim;
        bool is_row_major;
        Shape shape;

        BasicArrayRef(Ptr data, Tag tag, Dim ndim, bool rm, Shape shape)
            : data(data), tag(tag), ndim(ndim), is_row_major(rm), shape(shape) {}

        BasicArrayRef(Ptr data, Tag tag, Dim ndim, bool rm, idx_t x, idx_t y = 0, idx_t z = 0)
            : BasicArrayRef(data, tag, ndim, rm, {{x, y, z}}) {}

        template<class T>
        BasicArrayRef(T* data, Dim ndim, bool rm, idx_t x, idx_t y = 0, idx_t z = 0)
            : BasicArrayRef(data, get_tag<std14::remove_const_t<T>>(), ndim, rm, x, y, z) {}

        /// Non-const array can be converted to const, but not the other way around
        BasicArrayRef(BasicArrayRef<false> const& a)
            : BasicArrayRef(a.data, a.tag, a.ndim, a.is_row_major, a.shape) {}

        idx_t size() const { return std::accumulate(shape.begin(), shape.begin() + ndim,
                                                    idx_t{1}, std::multiplies<idx_t>()); }
    };

    template<class T>
    using MakeArrayRef = BasicArrayRef<std::is_const<std14::remove_pointer_t<T>>::value>;

    template<class T>
    auto make_arrayref(T* data, std::int8_t ndim, bool is_row_major, idx_t x, idx_t y, idx_t z)
                       -> MakeArrayRef<T> {
        return {data, get_tag<T>(), ndim, is_row_major, {{x, y, z}}};
    }
} // namespace detail

/// Const or mutable reference to any 1D, 2D or 3D array with any scalar type
using ArrayConstRef = detail::BasicArrayRef<true>;
using ArrayRef = detail::BasicArrayRef<false>;

/// Reference to an array which is limited to a set of scalar types
template<class Scalar, class... Scalars>
struct VariantArrayConstRef : ArrayConstRef {
    using First = Scalar;
    using Types = TypeList<Scalar, Scalars...>;

    VariantArrayConstRef(ArrayConstRef const& other) : ArrayConstRef(other) { check(other.tag); }
    VariantArrayConstRef(ArrayRef const& other) : ArrayConstRef(other) { check(other.tag); }

    void check(Tag candidate_tag) const {
        auto const possible_tags = {detail::get_tag<Scalar>(), detail::get_tag<Scalars>()...};
        auto const is_invalid = std::none_of(possible_tags.begin(), possible_tags.end(),
                                             [&](Tag tag) { return candidate_tag == tag; });
        if (is_invalid) {
            throw std::runtime_error("Invalid VariantArrayConstRef assignment");
        }
    }
};

template<class Scalar, class... Scalars>
struct VariantArrayRef : ArrayRef {
    using First = Scalar;
    using Types = TypeList<Scalar, Scalars...>;

    VariantArrayRef(ArrayRef const& other) : ArrayRef(other) {
        auto const possible_tags = {detail::get_tag<Scalar>(), detail::get_tag<Scalars>()...};
        auto const is_invalid = std::none_of(possible_tags.begin(), possible_tags.end(),
                                             [&](Tag tag) { return other.tag == tag; });
        if (is_invalid) {
            throw std::runtime_error("Invalid VariantArrayRef assignment");
        }
    }
};

/// Return a 1D array reference from pointer and size
template<class T>
auto arrayref(T* data, idx_t size) -> detail::MakeArrayRef<T> {
    return detail::make_arrayref(data, 1, true, size, 0, 0);
};

/// Return a 2D array reference from pointer and sizes
template<class T>
auto arrayref(T* data, idx_t x, idx_t y) -> detail::MakeArrayRef<T> {
    return detail::make_arrayref(data, 2, true, x, y, 0);
};

/// Return a 3D array reference from pointer and sizes
template<class T>
auto arrayref(T* data, idx_t x, idx_t y, idx_t z) -> detail::MakeArrayRef<T> {
    return detail::make_arrayref(data, 3, true, x, y, z);
};

// Common aliases
using RealArrayConstRef = VariantArrayConstRef<float, double>;
using ComplexArrayConstRef = VariantArrayConstRef<
    float, double, std::complex<float>, std::complex<double>
>;
using RealArrayRef = VariantArrayRef<float, double>;
using ComplexArrayRef = VariantArrayRef<
    float, double, std::complex<float>, std::complex<double>
>;

/**
 Creates an actual container from an ArrayRef

 To be specialized by concrete containers. The container can adopt the reference via
 a proxy type (e.g. Eigen::Map) or it can create a copy of the ArrayRef's data.
 */
template<class Container>
struct MakeContainer {
    // Intentionally unimplemented, specializations should do it
    static Container make(ArrayConstRef const&);
    static Container make(ArrayRef const&);
};

namespace detail {
    template<template<class> class Container, class Variant>
    using DeclContainer = decltype(
        MakeContainer<Container<typename Variant::First>>::make(std::declval<Variant>())
    );

    template<class Function, template<class> class Container, class Variant>
    using MatchResult = typename std::result_of<
        Function(DeclContainer<Container, Variant>)
    >::type;

    template<class Result, template<class> class /*Container*/, class Variant, class Function>
    Result try_match(Variant, Function, TypeList<>) {
        throw std::runtime_error{"A match was not found"};
    };

    template<class Result, template<class> class Container, class Variant, class Function,
             class Scalar, class... Tail>
    Result try_match(Variant ref, Function lambda, TypeList<Scalar, Tail...>) {
        if (ref.tag == detail::get_tag<Scalar>()) {
            return lambda(MakeContainer<Container<Scalar>>::make(ref));
        } else {
            return try_match<Result, Container>(ref, lambda, TypeList<Tail...>{});
        }
    };

    template<class Function, template<class> class Container1, template<class> class Container2,
             class Variant1, class Variant2>
    using Match2Result = typename std::result_of<
        Function(DeclContainer<Container1, Variant1>, DeclContainer<Container2, Variant2>)
    >::type;

    template<class Result, template<class> class /*Container1*/,
             template<class> class /*Container2*/, class Variant1, class Variant2, class Function>
    Result try_match2(Variant1, Variant2, Function, TypeList<>) {
        throw std::runtime_error{"A match was not found"};
    };

    template<class Result, template<class> class Container1, template<class> class Container2,
             class Variant1, class Variant2, class Function,
             class Scalar1, class Scalar2, class... Tail>
    Result try_match2(Variant1 ref1, Variant2 ref2, Function lambda,
                      TypeList<TypeList<Scalar1, Scalar2>, Tail...>) {
        if (ref1.tag == detail::get_tag<Scalar1>() && ref2.tag == detail::get_tag<Scalar2>()) {
            return lambda(MakeContainer<Container1<Scalar1>>::make(ref1),
                          MakeContainer<Container2<Scalar2>>::make(ref2));
        } else {
            return try_match2<Result, Container1, Container2>(ref1, ref2, lambda,
                                                              TypeList<Tail...>{});
        }
    };

    template<class List>
    struct IsSamePrecision;

    template<class T1, class T2>
    struct IsSamePrecision<TypeList<T1, T2>> {
        static constexpr auto value = std::is_same<
            num::get_real_t<T1>, num::get_real_t<T2>
        >::value;
    };
} // namespace detail

/// Match a VariantArrayRef to a Container and pass it to Function
template<template<class> class Container, class Variant, class Function,
         class Result = detail::MatchResult<Function, Container, Variant>>
Result match(Variant ref, Function lambda) {
    return detail::try_match<Result, Container>(ref, lambda, typename Variant::Types{});
}

/// Match two VariantArrayRefs to Containers (in all combinations) and pass them to Function
template<template<class> class Container1, template<class> class Container2,
         class Variant1, class Variant2, class Function,
         class Result = detail::Match2Result<Function, Container1, Container2, Variant1, Variant2>>
Result match2(Variant1 ref1, Variant2 ref2, Function lambda) {
    using List = tl::Combinations<typename Variant1::Types, typename Variant2::Types>;
    return detail::try_match2<Result, Container1, Container2>(ref1, ref2, lambda, List{});
}

/// Same as match2, but only considers matches where both scalar types have the same precision
template<template<class> class Container1, template<class> class Container2,
         class Variant1, class Variant2, class Function,
         class Result = detail::Match2Result<Function, Container1, Container2, Variant1, Variant2>>
Result match2sp(Variant1 ref1, Variant2 ref2, Function lambda) {
    using List = tl::Combinations<typename Variant1::Types, typename Variant2::Types>;
    using FilteredList = tl::Filter<List, detail::IsSamePrecision>;
    return detail::try_match2<Result, Container1, Container2>(ref1, ref2, lambda, FilteredList{});
}

}} // namespace cpb::num
