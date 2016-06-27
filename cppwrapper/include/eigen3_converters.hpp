#pragma once
#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"

#include <boost/python/to_python_converter.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/cast.hpp>
#include <boost/python/import.hpp>
#include <boost/python/object.hpp>
#include <boost/python/tuple.hpp>

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
    static constexpr auto is_const = std::is_base_of<cpb::num::ArrayConstRef, ArrayVariant>::value;
    static constexpr auto base_flags = NPY_ARRAY_ALIGNED | (!is_const ? NPY_ARRAY_WRITEABLE : 0);

    static PyObject* convert(ArrayVariant const& ref) {
        auto const ndim = (ref.rows == 1 || ref.cols == 1) ? 1 : 2;

        npy_intp shape[2];
        if (ndim == 1) {
            shape[0] = ref.is_row_major ? ref.cols : ref.rows;
        } else {
            shape[0] = ref.rows;
            shape[1] = ref.cols;
        }

        using cpb::num::Tag;
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

        int flags = base_flags | (ref.is_row_major ? NPY_ARRAY_C_CONTIGUOUS
                                                   : NPY_ARRAY_F_CONTIGUOUS);

        // ndarray from existing data -> it does not own the data and will not delete it
        return PyArray_New(&PyArray_Type, ndim, shape, type, nullptr,
                           const_cast<void*>(ref.data), 0, flags, nullptr);
    }

    static const PyTypeObject* get_pytype() { return &PyArray_Type; }
};

template<class T>
inline void register_arrayref_converter() {
    bp::to_python_converter<T, arrayref_to_python<T>>();
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
        auto ndarray = bp::handle<PyArrayObject>(bp::allow_null(PyArray_FROMANY(
            p, ndtype, ndim, ndim, NPY_ARRAY_FORCECAST |
            (EigenType::IsRowMajor ? NPY_ARRAY_C_CONTIGUOUS : NPY_ARRAY_F_CONTIGUOUS)
        )));

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

        auto ndarray = bp::handle<PyArrayObject>(PyArray_FROMANY(
            p, ndtype, ndim, ndim, NPY_ARRAY_FORCECAST |
            (EigenType::IsRowMajor ? NPY_ARRAY_C_CONTIGUOUS : NPY_ARRAY_F_CONTIGUOUS)
        ));
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
    bp::to_python_converter<EigenType, eigen3_to_numpy<EigenType>>();
}

template<class EigenType>
inline void extract_array(EigenType& v, bp::object const& o) {
    bp::extract<Eigen::Map<EigenType>> map{o};
    if (map.check()) {
        v = map();
    } else {
        v = bp::extract<EigenType>(o)();
    }
}

struct ExtractArray {
    bp::object result;

    template<class Array>
    void operator()(Eigen::Map<Array> ret) const {
        bp::extract<Eigen::Map<Array>> extract_map(result);
        if (extract_map.check()) {
            ret = extract_map();
        } else {
            ret = bp::extract<Array>(result)();
        }
    }
};

namespace detail {
    template<class scalar_t>
    void copy_data(bp::object from, int size, scalar_t* to) {
        using Array = cpb::ArrayX<scalar_t>;
        bp::extract<Eigen::Map<Array>> extract_map(from);
        if (extract_map.check()) {
            std::copy_n(extract_map().data(), size, to);
        } else {
            std::copy_n(bp::extract<Array>(from)().data(), size, to);
        }
    }
}

template<class scalar_t>
struct csr_eigen3_to_scipy {
    static PyObject* convert(cpb::SparseMatrixX<scalar_t> const& s) {
        auto scipy_sparse = bp::import("scipy.sparse");
        auto csr_matrix = scipy_sparse.attr("csr_matrix");

        auto data = cpb::arrayref(s.valuePtr(), s.nonZeros());
        auto indices = cpb::arrayref(s.innerIndexPtr(), s.nonZeros());
        auto indptr = cpb::arrayref(s.outerIndexPtr(), s.outerSize() + 1);
        auto matrix = csr_matrix(bp::make_tuple(data, indices, indptr),
                                 bp::make_tuple(s.rows(), s.cols()),
                                 /*dtype*/bp::object{}, /*copy*/bp::object{false});
        return matrix.release();
    }
};

template<class CsrRef>
struct csrref_to_scipy {
    static PyObject* convert(CsrRef const& s) {
        auto scipy_sparse = bp::import("scipy.sparse");
        auto csr_matrix = scipy_sparse.attr("csr_matrix");

        auto matrix = csr_matrix(bp::make_tuple(s.data_ref(), s.indices_ref(), s.indptr_ref()),
                                 bp::make_tuple(s.rows, s.cols),
                                 /*dtype*/bp::object{}, /*copy*/bp::object{false});
        return matrix.release();
    }
};

template<class T>
inline void register_csrref_converter() {
    bp::to_python_converter<T, csrref_to_scipy<T>>();
}

template<class scalar_t>
struct scipy_sparse_to_eigen3 {
    using SparseMatrix = cpb::SparseMatrixX<scalar_t>;

    scipy_sparse_to_eigen3() {
        bp::converter::registry::insert_rvalue_converter(
            &convertible, &construct, bp::type_id<SparseMatrix>()
        );
    }

    static void* convertible(PyObject* p) {
        auto o = bp::object(bp::handle<>(bp::borrowed(p)));
        auto type = bp::getattr(o, "dtype", {});
        if (type.is_none() || bp::extract<int>(type.attr("num")) != dtype<scalar_t>::value) {
            return nullptr;
        }

        if (bp::getattr(o, "shape", {}).is_none()) return nullptr;
        if (bp::getattr(o, "nnz", {}).is_none()) return nullptr;
        if (bp::getattr(o, "data", {}).is_none()) return nullptr;
        if (bp::getattr(o, "indices", {}).is_none()) return nullptr;
        if (bp::getattr(o, "indptr", {}).is_none()) return nullptr;

        return p;
    }

    static void construct(PyObject* p, bp::converter::rvalue_from_python_stage1_data* d) {
        auto storage = reinterpret_cast<SparseMatrix*>(
            ((bp::converter::rvalue_from_python_storage<SparseMatrix>*)d)->storage.bytes
        );

        auto pymatrix = bp::object(bp::handle<>(bp::borrowed(p)));
        auto shape = bp::getattr(pymatrix, "shape");
        auto const rows = bp::extract<int>(shape[0])();
        auto const cols = bp::extract<int>(shape[1])();

        new (storage) SparseMatrix(rows, cols);
        auto& sm = *storage;

        auto const nnz = bp::extract<int>(bp::getattr(pymatrix, "nnz"))();
        sm.resizeNonZeros(nnz);
        ::detail::copy_data(bp::getattr(pymatrix, "data"), nnz, sm.valuePtr());
        ::detail::copy_data(bp::getattr(pymatrix, "indices"), nnz, sm.innerIndexPtr());
        ::detail::copy_data(bp::getattr(pymatrix, "indptr"), rows + 1, sm.outerIndexPtr());

        d->convertible = storage;
    }
};

template<class scalar_t>
inline void register_csr_converter() {
    scipy_sparse_to_eigen3<scalar_t>{};
    bp::to_python_converter<cpb::SparseMatrixX<scalar_t>, csr_eigen3_to_scipy<scalar_t>>();
}
