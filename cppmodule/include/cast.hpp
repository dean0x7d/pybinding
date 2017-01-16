#pragma once
#include "numeric/arrayref.hpp"
#include "numeric/sparseref.hpp"
#include "support/variant.hpp"

namespace pybind11 { namespace detail {

/// Only for statically sized vectors: accepts smaller source arrays and zero-fills the remainder
template<class Vector>
struct static_vec_caster {
    using Scalar = typename Vector::Scalar;
    using Strides = Eigen::InnerStride<>;
    using Map = Eigen::Map<Eigen::Matrix<Scalar, 1, Eigen::Dynamic> const, 0, Strides>;

    bool load(handle src, bool) {
        auto const a = array_t<Scalar>::ensure(src);
        if (!a) { return false; }

        auto const ndim = a.ndim();
        if (ndim > 1) { return false; }

        value = Vector::Zero();
        if (ndim == 0) {
            value[0] = a.data()[0];
        } else {
            auto const size = static_cast<int>(a.size());
            if (size > Vector::SizeAtCompileTime) { return false; }

            auto const stride = static_cast<Eigen::Index>(a.strides()[0] / sizeof(Scalar));
            value.head(size) = Map(a.data(), size, Strides(stride));
        }
        return true;
    }

    static handle cast(Vector const& src, return_value_policy, handle /*parent*/) {
        return array_t<Scalar>(src.size(), src.data()).release();
    }

    PYBIND11_TYPE_CASTER(Vector, _("numpy.ndarray"));
};

template<> struct type_caster<Eigen::Vector3f> : static_vec_caster<Eigen::Vector3f> {};
template<> struct type_caster<Eigen::Vector3i> : static_vec_caster<Eigen::Vector3i> {};

template<bool is_const>
struct arrayref_caster {
    using Type = cpb::num::BasicArrayRef<is_const>;
    using Shape = std::array<Py_intptr_t, 2>;
    static constexpr auto writable = !is_const ? npy_api::NPY_ARRAY_WRITEABLE_ : 0;
    static constexpr auto base_flags = npy_api::NPY_ARRAY_ALIGNED_ | writable;

    static handle cast(Type const& src, return_value_policy, handle parent) {
        using cpb::num::Tag;
        auto data_type = [&] {
            switch (src.tag) {
                case Tag::f32:  return dtype::of<float>();
                case Tag::cf32: return dtype::of<std::complex<float>>();
                case Tag::f64:  return dtype::of<double>();
                case Tag::cf64: return dtype::of<std::complex<double>>();
                case Tag::b:    return dtype::of<bool>();
                case Tag::i8:   return dtype::of<int8_t>();
                case Tag::i16:  return dtype::of<int16_t>();
                case Tag::i32:  return dtype::of<int32_t>();
                case Tag::i64:  return dtype::of<int64_t>();
                case Tag::u8:   return dtype::of<uint8_t>();
                case Tag::u16:  return dtype::of<uint16_t>();
                case Tag::u32:  return dtype::of<uint32_t>();
                case Tag::u64:  return dtype::of<uint64_t>();
                default: throw std::runtime_error("ArrayRef: unknown scalar type");
            }
        }();

        auto const ndim = (src.rows == 1 || src.cols == 1) ? 1 : 2;
        auto shape = (ndim == 1) ? Shape{{src.is_row_major ? src.cols : src.rows}}
                                 : Shape{{src.rows, src.cols}};
        auto const flags = base_flags | (src.is_row_major ? npy_api::NPY_C_CONTIGUOUS_
                                                          : npy_api::NPY_F_CONTIGUOUS_);

        auto result = npy_api::get().PyArray_NewFromDescr_(
            npy_api::get().PyArray_Type_, data_type.release().ptr(), ndim, shape.data(),
            /*strides*/nullptr, const_cast<void*>(src.data), flags, nullptr
        );
        if (!result) { pybind11_fail("ArrayRef: unable to create array"); }
        if (parent) { detail::keep_alive_impl(result, parent); }
        return result;
    }

    static PYBIND11_DESCR name() { return type_descr(_("numpy.ndarray")); }
};

template<bool is_const>
struct type_caster<cpb::num::BasicArrayRef<is_const>> : arrayref_caster<is_const> {};
template<class T, class... Ts>
struct type_caster<cpb::num::VariantArrayConstRef<T, Ts...>> : arrayref_caster<true> {};
template<class T, class... Ts>
struct type_caster<cpb::num::VariantArrayRef<T, Ts...>> : arrayref_caster<false> {};

struct csrref_caster {
    using Type = cpb::num::AnyCsrConstRef;

    static handle cast(Type const& src, return_value_policy policy, handle parent) {
        auto const data = pybind11::cast(src.data_ref(), policy, parent);
        auto const indices = pybind11::cast(src.indices_ref(), policy, parent);
        auto const indptr = pybind11::cast(src.indptr_ref(), policy, parent);

        auto result = module::import("scipy.sparse").attr("csr_matrix")(
            pybind11::make_tuple(data, indices, indptr),
            pybind11::make_tuple(src.rows, src.cols),
            /*dtype*/none(), /*copy*/false
        );
        if (parent) { detail::keep_alive_impl(result, parent); }
        return result.release();
    }

    static PYBIND11_DESCR name() { return type_descr(_("scipy.sparse.csr_matrix")); }
};

template<> struct type_caster<cpb::num::AnyCsrConstRef> : csrref_caster {};
template<class T, class... Ts>
struct type_caster<cpb::num::VariantCsrConstRef<T, Ts...>> : csrref_caster {};
template<class T>
struct type_caster<cpb::num::CsrConstRef<T>> : csrref_caster {};

struct variant_visitor {
    return_value_policy policy;
    handle parent;

    template<class T>
    handle operator()(T const& src) const {
        return make_caster<T>::cast(src, policy, parent);
    }
};

template<class Variant> struct variant_caster;

template<template<class...> class V, class... Ts>
struct variant_caster<V<Ts...>> {
    using Type = V<Ts...>;

    template<class T>
    bool load_one(handle src, bool convert) {
        auto caster = make_caster<T>();
        if (caster.load(src, convert)) {
            value = cast_op<T>(caster);
            return true;
        }
        return false;
    }

    bool load(handle src, bool convert) {
        auto loaded = {false, load_one<Ts>(src, convert)...};
        return std::any_of(loaded.begin(), loaded.end(), [](bool b) { return b; });
    }

    static handle cast(Type const& src, return_value_policy policy, handle parent) {
        return cpb::var::apply_visitor(variant_visitor{policy, parent}, src);
    }

    PYBIND11_TYPE_CASTER(Type, _("Variant"));
};

template<class... Args> struct type_caster<cpb::var::variant<Args...>>
    : variant_caster<cpb::var::variant<Args...>> {};

}} // namespace pybind11::detail
