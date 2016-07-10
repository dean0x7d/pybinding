#pragma once
#include "numeric/arrayref.hpp"
#include "numeric/sparseref.hpp"

#include <pybind11/cast.h>
#include <pybind11/numpy.h>

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace pybind11 { namespace detail {

// TODO: Remove this in favor of an explicit conversion in Python
template<class T>
struct eigen_compiletime_vector_caster {
    using Scalar = typename T::Scalar;
    static constexpr bool is_row_major = T::Flags & Eigen::RowMajorBit;

    bool load(handle src, bool) {
        auto buffer = array_t<Scalar>(src, true);
        if (!buffer.check()) {
            return false;
        }

        auto const info = buffer.request();
        if (info.ndim != 1) {
            return false;
        }

        using Strides = Eigen::InnerStride<>;
        auto const strides = Strides(info.strides[0] / sizeof(Scalar));
        auto const size = static_cast<Strides::Index>(info.shape[0]);
        if (size > T::SizeAtCompileTime) {
            return false;
        }
        
        value = T::Zero();
        value.head(size) = Eigen::Map<Eigen::Matrix<Scalar, -1, -1>, 0, Strides>(
            static_cast<Scalar*>(info.ptr),
            is_row_major ? 1 : size,
            is_row_major ? size : 1,
            strides
        );

        return true;
    }

    static handle cast(T const& src, return_value_policy /* policy */, handle /* parent */) {
        return array_t<Scalar>(src.size(), src.data()).release();
    }

    PYBIND11_TYPE_CASTER(T, _("numpy.ndarray[") + npy_format_descriptor<Scalar>::name()
        + _("[") + _<T::RowsAtCompileTime>() + _(", ") + _<T::ColsAtCompileTime>() + _("]]"));
};

template<>
struct type_caster<Eigen::Vector3f> : eigen_compiletime_vector_caster<Eigen::Vector3f> {};
template<>
struct type_caster<Eigen::Vector3i> : eigen_compiletime_vector_caster<Eigen::Vector3i> {};


template<bool is_const>
struct type_caster<cpb::num::BasicArrayRef<is_const>> {
    using Type = cpb::num::BasicArrayRef<is_const>;
    using Shape = std::array<Py_intptr_t, 2>;

    static constexpr auto base_flags = npy_api::NPY_ARRAY_ALIGNED_
                                       | (!is_const ? npy_api::NPY_ARRAY_WRITEABLE_ : 0);

    [[noreturn]] bool load(handle, bool) {
        pybind11_fail("An ArrayRef cannot be created from a Python object");
    }

    static handle cast(Type const& src, return_value_policy /*policy*/, handle parent) {
        using cpb::num::Tag;
        auto data_type = [&]() {
            switch (src.tag) {
                case Tag::f32:  return dtype::of<float>();
                case Tag::cf32: return dtype::of<std::complex<float>>();
                case Tag::f64:  return dtype::of<double>();
                case Tag::cf64: return dtype::of<std::complex<double>>();
                case Tag::b:    return dtype::of<bool>();
                case Tag::i8:   return dtype::of<int8_t>();
                case Tag::i16:  return dtype::of<int16_t>();
                case Tag::i32:  return dtype::of<int32_t>();
                case Tag::u8:   return dtype::of<uint8_t>();
                case Tag::u16:  return dtype::of<uint16_t>();
                case Tag::u32:  return dtype::of<uint32_t>();
                default: return dtype();
            }
        }();

        auto const ndim = (src.rows == 1 || src.cols == 1) ? 1 : 2;
        auto shape = (ndim == 1) ? Shape{{src.is_row_major ? src.cols : src.rows}}
                                 : Shape{{src.rows, src.cols}};
        auto const flags = base_flags | (src.is_row_major ? npy_api::NPY_C_CONTIGUOUS_
                                                          : npy_api::NPY_F_CONTIGUOUS_);

        auto result = npy_api::get().PyArray_NewFromDescr_(
            npy_api::get().PyArray_Type_, data_type.release().ptr(), ndim, shape.data(),
            /*strides*/nullptr, const_cast<void *>(src.data), flags, nullptr
        );
        if (!result) {
            pybind11_fail("ArrayRef: unable to create array");
        }

        if (parent) {
            detail::keep_alive_impl(result, parent);
        }

        return result;
    }

    PYBIND11_TYPE_CASTER(Type, _("numpy.ndarray"));
};

template<class T, class... Ts>
struct type_caster<cpb::num::VariantArrayConstRef<T, Ts...>>
    : type_caster<cpb::num::BasicArrayRef<true>> {};

template<class T, class... Ts>
struct type_caster<cpb::num::VariantArrayRef<T, Ts...>>
    : type_caster<cpb::num::BasicArrayRef<false>> {};

template<>
struct type_caster<cpb::num::AnyCsrConstRef> {
    using Type = cpb::num::AnyCsrConstRef;

    [[noreturn]] bool load(handle, bool) {
        pybind11_fail("A CsrConstRef cannot be created from a Python object");
    }

    static handle cast(Type const& src, return_value_policy policy, handle parent) {
        if (parent) {
            policy = return_value_policy::reference_internal;
        }

        auto const data = pybind11::cast(src.data_ref(), policy, parent);
        auto const indices = pybind11::cast(src.indices_ref(), policy, parent);
        auto const indptr = pybind11::cast(src.indptr_ref(), policy, parent);

        auto result = module::import("scipy.sparse").attr("csr_matrix")(
            pybind11::make_tuple(data, indices, indptr),
            pybind11::make_tuple(src.rows, src.cols),
            /*dtype*/none(), /*copy*/false
        );
        if (parent) {
            detail::keep_alive_impl(result, parent);
        }
        return result.release();
    }

    PYBIND11_TYPE_CASTER(Type, _("scipy.sparse.csr_matrix"));
};

template<class T, class... Ts>
struct type_caster<cpb::num::VariantCsrConstRef<T, Ts...>>
    : type_caster<cpb::num::AnyCsrConstRef> {};

template<class T>
struct type_caster<cpb::num::CsrConstRef<T>>
    : type_caster<cpb::num::AnyCsrConstRef> {};

}} // namespace pybind11::detail
