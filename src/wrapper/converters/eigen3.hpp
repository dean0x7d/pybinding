#pragma once
#include <boost/python/to_python_converter.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/borrowed.hpp>
#include <boost/python/cast.hpp>
namespace bp = boost::python;

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include <numpy/ndarrayobject.h>
#include "support/uref.hpp"
#include <cstdint>

/// type map: C++ type to numpy type
template<class T>
struct numpy_type_map {
    static const int typenum;
};
template<> const int numpy_type_map<bool>::typenum = NPY_BOOL;
template<> const int numpy_type_map<float>::typenum = NPY_FLOAT;
template<> const int numpy_type_map<double>::typenum = NPY_DOUBLE;
template<> const int numpy_type_map<std::complex<float> >::typenum = NPY_CFLOAT;
template<> const int numpy_type_map<std::complex<double> >::typenum = NPY_CDOUBLE;
template<> const int numpy_type_map<std::int8_t>::typenum = NPY_INT8;
template<> const int numpy_type_map<std::uint8_t>::typenum = NPY_UINT8;
template<> const int numpy_type_map<std::int16_t>::typenum = NPY_INT16;
template<> const int numpy_type_map<std::uint16_t>::typenum = NPY_UINT16;
template<> const int numpy_type_map<std::int32_t>::typenum = NPY_INT32;
template<> const int numpy_type_map<std::uint32_t>::typenum = NPY_UINT32;
template<> const int numpy_type_map<std::int64_t>::typenum = NPY_INT64;
template<> const int numpy_type_map<std::uint64_t>::typenum = NPY_UINT64;

struct denseuref_to_python {
    static PyObject* convert(const DenseURef& u) {
        int ndim = (u.rows == 1 || u.cols == 1) ? 1 : 2;

        npy_intp shape[ndim];
        if (ndim == 1) { // row or column vector
            shape[0] = u.is_row_major ? u.cols : u.rows;
        }
        else { // matrix
            shape[0] = u.rows;
            shape[1] = u.cols;
        }

        int type = NPY_INT32;
        if (u.type == ScalarType::f)  type = NPY_FLOAT;
        if (u.type == ScalarType::cf) type = NPY_CFLOAT;
        if (u.type == ScalarType::d)  type = NPY_DOUBLE;
        if (u.type == ScalarType::cd) type = NPY_CDOUBLE;

        int flags = u.is_row_major ? NPY_ARRAY_CARRAY : NPY_ARRAY_FARRAY;

        // ndarray from existing data -> it does not own the data and will not delete it
        return PyArray_New(&PyArray_Type, ndim, shape, type, nullptr,
                           (void*)u.data, 0, flags, nullptr);
    }

    static const PyTypeObject* get_pytype() { return &PyArray_Type; }
};

template<class EigenType>
struct eigen3_to_numpy {
    static PyObject* convert(const EigenType& eigen_object) {
        using scalar_t = typename EigenType::Scalar;
        constexpr int ndim = EigenType::IsVectorAtCompileTime ? 1 : 2;

        npy_intp shape[ndim];
        if (ndim == 1) { // row or column vector
            shape[0] = EigenType::IsRowMajor ? eigen_object.cols() : eigen_object.rows();
        }
        else { // matrix
            shape[0] = eigen_object.rows();
            shape[1] = eigen_object.cols();
        }
        
        int type = numpy_type_map<scalar_t>::typenum;
        int flags = EigenType::IsRowMajor ? NPY_ARRAY_CARRAY : NPY_ARRAY_FARRAY;

        // new empty ndarray of the correct type and shape
        PyObject* array = PyArray_New(&PyArray_Type, ndim, shape, type, nullptr,
                                      nullptr, 0, flags, nullptr);
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
template<class EigenType, int NDIM>
struct construct_eigen {
    inline static void func(void* storage, PyArrayObject* ndarray, npy_intp* shape) {
        using scalar_t = typename EigenType::Scalar;
        new (storage) EigenType(static_cast<scalar_t*>(PyArray_DATA(ndarray)), shape[0]);
    }
};

template<class EigenType>
struct construct_eigen<EigenType, 2> {
    inline static void func(void* storage, PyArrayObject* ndarray, npy_intp* shape) {
        using scalar_t = typename EigenType::Scalar;
        new (storage) EigenType(static_cast<scalar_t*>(PyArray_DATA(ndarray)), shape[0], shape[1]);
    }
};

template<class EigenType>
struct numpy_to_eigen3 {
    numpy_to_eigen3() {
        bp::converter::registry::push_back(&convertible, &construct, bp::type_id<EigenType>());
    }
    
    static void* convertible(PyObject* obj_ptr) {
        if (!PyArray_Check(obj_ptr))
            return nullptr;
        
        constexpr int ndim = EigenType::IsVectorAtCompileTime ? 1 : 2;
        int type = numpy_type_map<typename EigenType::Scalar>::typenum;

        // try to make an ndarray from the python object
        PyArrayObject* ndarray = (PyArrayObject*)PyArray_FromObject(obj_ptr, type, ndim, ndim);
        
        bool is_convertible = true;
        // not possible, fail
        if (!ndarray)
            is_convertible = false;
        
        if (is_convertible && ndim == 2)
        { // additional check for matrices
            if (EigenType::IsRowMajor && !PyArray_IS_C_CONTIGUOUS(ndarray))
                is_convertible = false; // row major only accepts C-style array
            if (!EigenType::IsRowMajor && !PyArray_IS_F_CONTIGUOUS(ndarray))
                is_convertible = false; // column major only accepts Fortran-style array
        }

        // we don't really need the array object, remove the reference
        Py_XDECREF(ndarray);
        
        return is_convertible ? obj_ptr : nullptr;
    }
    
    static void construct(PyObject* obj_ptr, bp::converter::rvalue_from_python_stage1_data* data) {
        // get the pointer to memory where to construct the new eigen3 object
        void* storage = ((bp::converter::rvalue_from_python_storage<EigenType>*)
                         data)->storage.bytes;
     
        constexpr int ndim = EigenType::IsVectorAtCompileTime ? 1 : 2;
        int type = numpy_type_map<typename EigenType::Scalar>::typenum;
        
        PyArrayObject* ndarray = (PyArrayObject*)PyArray_FromObject(obj_ptr, type, ndim, ndim);
        npy_intp* shape = PyArray_DIMS(ndarray);

        // in-place construct a new eigen3 object using data from the numpy array
        construct_eigen<EigenType, ndim>::func(storage, ndarray, shape);
        
        // save the pointer to the eigen3 object for later use by boost.python
        data->convertible = storage;

        Py_DECREF(ndarray);
    }
};

/**
 boost_python converter: python tuple to eigen3 type
 */
template<class EigenType, bool is_vector>
struct tuple_to_eigen3 {
    tuple_to_eigen3() {
        bp::converter::registry::push_back(&convertible, &construct, bp::type_id<EigenType>());
    }
    
    static void* convertible(PyObject* obj_ptr) {
        constexpr int ndim = EigenType::IsVectorAtCompileTime ? 1 : 2;
        static_assert(ndim == 1, "Only 1D arrays may be extracted from tuples.");
        
        // expecting a tuple
        if (!PyTuple_CheckExact(obj_ptr))
            return nullptr;
        
        bp::tuple tup{bp::borrowed(obj_ptr)};
        // if the Eigen array is fixed size, the tuple can't be bigger than it
        if (EigenType::SizeAtCompileTime > 0 && bp::len(tup) > EigenType::SizeAtCompileTime)
            return nullptr;
        
        // make sure the scalar type is compatible
        bp::extract<typename EigenType::Scalar> ex_scalar(tup[0]);
        if (!ex_scalar.check())
            return nullptr;
        
        return obj_ptr;
    }
    
    static void construct(PyObject* obj_ptr, bp::converter::rvalue_from_python_stage1_data* data) {
        // get the pointer to memory where to construct the new eigen3 object
        void* storage = ((bp::converter::rvalue_from_python_storage<EigenType>*)
                         data)->storage.bytes;
        
        bp::tuple tup{bp::borrowed(obj_ptr)};
        
        // in-place construct a new eigen3 object using data from the numpy array
        if (EigenType::SizeAtCompileTime > 0)
            new (storage) EigenType();
        else
            new (storage) EigenType(bp::len(tup));

        EigenType& et = *(EigenType*)storage;

        // Vector3 may be converted from (x, y) in which case it's read as (x, y, 0)
        if (EigenType::SizeAtCompileTime > 0)
            et.setZero();

        for (int i = 0; i < bp::len(tup); i++)
            et(i) = bp::extract<typename EigenType::Scalar>(tup[i]);
        
        // save the pointer to the eigen3 object for later use by boost.python
        data->convertible = storage;
    }
};

// enable tuple converter only for 1D arrays
template<class EigenType> struct tuple_to_eigen3<EigenType, false> {};

// suppress unused variable warnings
template<class T> inline void force_instantiate(const T&) {}

template<class EigenType>
inline void eigen3_numpy_register_type() {
    using namespace Eigen;
    using namespace boost::python;
    
    to_python_converter<Ref<const EigenType>, eigen3_to_numpy<Ref<const EigenType>>, true>{};
    to_python_converter<Ref<EigenType>, eigen3_to_numpy<Ref<EigenType>>, true>{};
    to_python_converter<Map<EigenType>, eigen3_to_numpy<Map<EigenType>>, true>{};
    to_python_converter<EigenType, eigen3_to_numpy<EigenType>, true>{};

    force_instantiate(numpy_to_eigen3<Map<EigenType>>{});
    implicitly_convertible<Map<EigenType>, EigenType>();
    implicitly_convertible<Map<EigenType>, Ref<EigenType>>();
    implicitly_convertible<Map<EigenType>, Ref<const EigenType>>();

    force_instantiate(tuple_to_eigen3<EigenType, EigenType::IsVectorAtCompileTime>{});
}
