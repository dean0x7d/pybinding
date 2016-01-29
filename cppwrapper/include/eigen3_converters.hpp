#pragma once
#include "support/dense.hpp"

#include <boost/python/to_python_converter.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/cast.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include <numpy/ndarrayobject.h>

#include <cstdint>

namespace bp = boost::python;


/// type map: C++ type to numpy type
template<class T> struct dtype;
template<> struct dtype<bool> { static constexpr auto value = NPY_BOOL; };
template<> struct dtype<float> { static constexpr auto value = NPY_FLOAT; };
template<> struct dtype<double> { static constexpr auto value = NPY_DOUBLE; };
template<> struct dtype<std::complex<float>> { static constexpr auto value = NPY_CFLOAT; };
template<> struct dtype<std::complex<double>> { static constexpr auto value = NPY_CDOUBLE; };
template<> struct dtype<std::int8_t> { static constexpr auto value = NPY_INT8; };
template<> struct dtype<std::int16_t> { static constexpr auto value = NPY_INT16; };
template<> struct dtype<std::int32_t> { static constexpr auto value = NPY_INT32; };
template<> struct dtype<std::int64_t> { static constexpr auto value = NPY_INT64; };
template<> struct dtype<std::uint8_t> { static constexpr auto value = NPY_UINT8; };
template<> struct dtype<std::uint16_t> { static constexpr auto value = NPY_UINT16; };
template<> struct dtype<std::uint32_t> { static constexpr auto value = NPY_UINT32; };
template<> struct dtype<std::uint64_t> { static constexpr auto value = NPY_UINT64; };

namespace boost { namespace python {
    template<> struct base_type_traits<PyArrayObject> : std::true_type {};
}}

template<class ArrayVariant>
struct arrayref_to_python {
    static PyObject* convert(ArrayVariant const& ref) {
        auto const ndim = (ref.rows == 1 || ref.cols == 1) ? 1 : 2;

        npy_intp shape[2];
        if (ndim == 1) {
            shape[0] = ref.is_row_major ? ref.cols : ref.rows;
        } else {
            shape[0] = ref.rows;
            shape[1] = ref.cols;
        }

        using tbm::num::Tag;
        auto const type = [&]{
            switch (ref.tag) {
                case Tag::f32:  return NPY_FLOAT;
                case Tag::cf32: return NPY_CFLOAT;
                case Tag::f64:  return NPY_DOUBLE;
                case Tag::cf64: return NPY_CDOUBLE;
                case Tag::b:    return NPY_BOOL;
                case Tag::i8:   return NPY_INT8;
                case Tag::i16:  return NPY_INT16;
                case Tag::i32:  return NPY_INT32;
                case Tag::u8:   return NPY_UINT8;
                case Tag::u16:  return NPY_UINT16;
                case Tag::u32:  return NPY_UINT32;
                default: return NPY_VOID;
            }
        }();

        int flags = ref.is_row_major ? NPY_ARRAY_CARRAY : NPY_ARRAY_FARRAY;

        // ndarray from existing data -> it does not own the data and will not delete it
        return PyArray_New(&PyArray_Type, ndim, shape, type, nullptr,
                           const_cast<void*>(ref.data), 0, flags, nullptr);
    }

    static const PyTypeObject* get_pytype() { return &PyArray_Type; }
};

template<class T>
inline void register_arrayref_converter() {
    bp::to_python_converter<T, arrayref_to_python<T>>{};
}

template<class EigenType>
struct eigen3_to_numpy {
    static PyObject* convert(const EigenType& eigen_object) {
        constexpr int ndim = EigenType::IsVectorAtCompileTime ? 1 : 2;

        npy_intp shape[ndim];
        if (ndim == 1) { // row or column vector
            shape[0] = EigenType::IsRowMajor ? eigen_object.cols() : eigen_object.rows();
        }
        else { // matrix
            shape[0] = eigen_object.rows();
            shape[1] = eigen_object.cols();
        }

        using scalar_t = typename EigenType::Scalar;
        int flags = EigenType::IsRowMajor ? NPY_ARRAY_CARRAY : NPY_ARRAY_FARRAY;

        // new empty ndarray of the correct type and shape
        PyObject* array = PyArray_New(&PyArray_Type, ndim, shape, dtype<scalar_t>::value,
                                      nullptr, nullptr, 0, flags, nullptr);
        std::memcpy(
            PyArray_DATA(bp::downcast<PyArrayObject>(array)),
            eigen_object.data(),
            eigen_object.size() * sizeof(scalar_t)
        );
        return array;
    }

    static const PyTypeObject* get_pytype() { return &PyArray_Type; }
};

/**
 Helper function that will construct an eigen vector or matrix
 */
template<class EigenType, int ndim, bool fixed_size> struct construct_eigen;

template<class EigenType>
struct construct_eigen<EigenType, 1, false> {
    static void exec(EigenType* v, typename EigenType::Scalar* data, npy_intp const* shape) {
        new (v) EigenType(shape[0]);
        std::copy_n(data, v->size(), v->data());
    }
};

template<class EigenType>
struct construct_eigen<EigenType, 2, false> {
    static void exec(EigenType* v, typename EigenType::Scalar* data, npy_intp const* shape) {
        new (v) EigenType(shape[0], shape[1]);
        std::copy_n(data, v->rows() * v->cols(), v->data());
    }
};

template<class EigenType>
struct construct_eigen<EigenType, 1, true> {
    static void exec(EigenType* v, typename EigenType::Scalar* data, npy_intp const* shape) {
        new (v) EigenType(EigenType::Zero());
        auto size = std::min(v->size(), shape[0]);
        std::copy_n(data, size, v->data());
    }
};


template<class EigenType>
struct numpy_to_eigen3 {
    numpy_to_eigen3() {
        bp::converter::registry::insert_rvalue_converter(
            &convertible, &construct, bp::type_id<EigenType>(), &PyArray_Type
        );
    }

    static constexpr auto ndim = EigenType::IsVectorAtCompileTime ? 1 : 2;
    static constexpr auto ndtype = dtype<typename EigenType::Scalar>::value;
    static constexpr auto fixed_size = EigenType::SizeAtCompileTime > 0;

    static void* convertible(PyObject* p) {
        // try to make an ndarray from the python object
        auto ndarray = bp::handle<PyArrayObject>{bp::allow_null(PyArray_FROMANY(
            p, ndtype, ndim, ndim, NPY_ARRAY_FORCECAST |
            (EigenType::IsRowMajor ? NPY_ARRAY_C_CONTIGUOUS : NPY_ARRAY_F_CONTIGUOUS)
        ))};

        if (!ndarray)
            return nullptr;
        if (EigenType::IsRowMajor && !PyArray_IS_C_CONTIGUOUS(ndarray.get()))
            return nullptr; // row major only accepts C-style array
        if (!EigenType::IsRowMajor && !PyArray_IS_F_CONTIGUOUS(ndarray.get()))
            return nullptr; // column major only accepts Fortran-style array

        return p;
    }

    static void construct(PyObject* p, bp::converter::rvalue_from_python_stage1_data* data) {
        // get the pointer to memory where to construct the new eigen3 object
        auto storage = reinterpret_cast<EigenType*>(
            ((bp::converter::rvalue_from_python_storage<EigenType>*)data)->storage.bytes
        );

        auto ndarray = bp::handle<PyArrayObject>{PyArray_FROMANY(
            p, ndtype, ndim, ndim, NPY_ARRAY_FORCECAST |
            (EigenType::IsRowMajor ? NPY_ARRAY_C_CONTIGUOUS : NPY_ARRAY_F_CONTIGUOUS)
        )};
        auto array_data = static_cast<typename EigenType::Scalar*>(PyArray_DATA(ndarray.get()));
        auto array_shape = PyArray_SHAPE(ndarray.get());

        // in-place construct a new eigen3 object using data from the numpy array
        construct_eigen<EigenType, ndim, fixed_size>::exec(storage, array_data, array_shape);

        // save the pointer to the eigen3 object for later use by boost.python
        data->convertible = storage;
    }
};

template<class EigenType, int ndim> struct construct_eigen_map;

template<class EigenType>
struct construct_eigen_map<EigenType, 1> {
    static void exec(void* storage, typename EigenType::Scalar* data, npy_intp const* shape) {
        new (storage) Eigen::Map<EigenType>{data, shape[0]};
    }
};

template<class EigenType>
struct construct_eigen_map<EigenType, 2> {
    static void exec(void* storage, typename EigenType::Scalar* data, npy_intp const* shape) {
        new (storage) Eigen::Map<EigenType>{data, shape[0], shape[1]};
    }
};

template<class EigenType>
struct numpy_to_eigen3_map {
    numpy_to_eigen3_map() {
        bp::converter::registry::insert_rvalue_converter(
            &convertible, &construct, bp::type_id<Eigen::Map<EigenType>>(), &PyArray_Type
        );
    }

    static constexpr auto ndim = EigenType::IsVectorAtCompileTime ? 1 : 2;
    static constexpr auto ndtype = dtype<typename EigenType::Scalar>::value;

    static void* convertible(PyObject* p) {
        if (!PyArray_Check(p))
            return nullptr;

        auto array = (PyArrayObject*)p;
        if (PyArray_NDIM(array) != ndim || PyArray_TYPE(array) != ndtype)
            return nullptr;
        if (EigenType::IsRowMajor && !PyArray_IS_C_CONTIGUOUS(array))
            return nullptr;
        if (!EigenType::IsRowMajor && !PyArray_IS_F_CONTIGUOUS(array))
            return nullptr;

        return p;
    }

    static void construct(PyObject* p, bp::converter::rvalue_from_python_stage1_data* data) {
        void* storage = ((bp::converter::rvalue_from_python_storage<EigenType>*)
            data)->storage.bytes;

        auto array = (PyArrayObject*)p;
        auto array_data = static_cast<typename EigenType::Scalar*>(PyArray_DATA(array));
        auto array_shape = PyArray_SHAPE(array);

        construct_eigen_map<EigenType, ndim>::exec(storage, array_data, array_shape);
        data->convertible = storage;
    }
};

template<class EigenType>
inline void eigen3_numpy_register_type() {
    numpy_to_eigen3<EigenType>{};
    numpy_to_eigen3_map<EigenType>{};
    bp::to_python_converter<EigenType, eigen3_to_numpy<EigenType>>{};
}

template<class EigenType>
inline void extract_array(EigenType& v, bp::object const& o) {
    bp::extract<Eigen::Map<EigenType>> map{o};
    if (map.check())
        v = map();
    else
        v = bp::extract<EigenType>{o}();
}
