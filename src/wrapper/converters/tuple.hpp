#pragma once

#include <boost/python/to_python_converter.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/borrowed.hpp>
namespace bp = boost::python;

template<int...> struct integer_sequence{};
template<int N, int... seq> struct make_sequence : make_sequence<N-1, N-1, seq...>{};
template<int... seq> struct make_sequence<0, seq...> { using type = integer_sequence<seq...>; };

/**
 boost_python converter: C++11 tuple to python tuple
 */
template <typename... Args>
struct to_bptuple {
    const std::tuple<Args...>& cpp_tuple;
    
    operator bp::tuple() const {
        return make_bptuple(typename make_sequence<sizeof...(Args)>::type());
    }
    
    template<int... S>
    bp::tuple make_bptuple(integer_sequence<S...>) const {
        return bp::make_tuple(std::get<S>(cpp_tuple)...);
    }
};

template<typename ... Args>
struct tuple_cpp_to_python {
    static PyObject* convert(const std::tuple<Args...>& cpp_tuple) {
        bp::tuple bp_tuple = to_bptuple<Args...>{cpp_tuple};
        return bp::incref(bp_tuple.ptr());
    }
};

/**
 boost_python converter: python tuple to C++11 tuple
 */
template <typename... Args>
struct extract_cpptuple {
    const bp::tuple& bp_tuple;
    
    operator std::tuple<Args...>() const {
        return make_cpptuple(typename make_sequence<sizeof...(Args)>::type());
    }
    
    template<int... S>
    std::tuple<Args...> make_cpptuple(integer_sequence<S...>) const {
        return std::make_tuple(static_cast<Args>(bp::extract<Args>(bp_tuple[S]))...);
    }
};

template<typename... Args>
struct tuple_python_to_cpp {
    tuple_python_to_cpp() {
        bp::converter::registry::push_back(&convertible, &construct,
                                           bp::type_id<std::tuple<Args...>>());
    }
    
    static void* convertible(PyObject* obj_ptr) {
        return PyTuple_CheckExact(obj_ptr) ? obj_ptr : nullptr;
    }
    
    static void construct(PyObject* obj_ptr, bp::converter::rvalue_from_python_stage1_data* data) {
        void* storage = ((bp::converter::rvalue_from_python_storage<std::tuple<Args...>>*)
                         data)->storage.bytes;
        bp::tuple tup{bp::borrowed(obj_ptr)};
        new (storage) std::tuple<Args...>(extract_cpptuple<Args...>{tup});
        data->convertible = storage;
    }
};

template<typename... Args>
void create_tuple_converter() {
    bp::to_python_converter<std::tuple<Args...>, tuple_cpp_to_python<Args...>>{};
    tuple_python_to_cpp<Args...>{};
}

#include <boost/python/list.hpp>
#include <vector>

template<typename T>
struct vector_to_list {
    static PyObject* convert(const std::vector<T>& v) {
        bp::list l;
        for (const auto& item : v){
            l.append(item);
        }
        return bp::incref(l.ptr());
    }
};

template<class T>
struct list_to_vector {
    list_to_vector() {
        bp::converter::registry::push_back(&convertible, &construct, bp::type_id<std::vector<T>>());
    }
    
    static void* convertible(PyObject* obj_ptr) {
        if (!PyList_CheckExact(obj_ptr))
            return nullptr;
        
        bp::list l{bp::borrowed(obj_ptr)};
        if (bp::len(l) >= 1) {
            bp::extract<T> ex{l[0]};
            if (!ex.check())
                return nullptr;
        }
        
        return obj_ptr;
    }
    
    static void construct(PyObject* obj_ptr, bp::converter::rvalue_from_python_stage1_data* data) {
        void* storage = ((bp::converter::rvalue_from_python_storage<std::vector<T>>*)
                         data)->storage.bytes;
        
        bp::tuple l{bp::borrowed(obj_ptr)};
        
        // in-place construct
        new (storage) std::vector<T>(bp::len(l));
        
        std::vector<T>& v = *(std::vector<T>*)storage;
        
        for (int i = 0; i < bp::len(l); ++i)
            v[i] = bp::extract<T>(l[i]);
        
        // save the pointer to the eigen3 object for later use by boost.python
        data->convertible = storage;
    }
};


template<typename T>
void create_vector_converter() {
    bp::to_python_converter<std::vector<T>, vector_to_list<T>>{};
    list_to_vector<T>{};
}

#include <boost/python/dict.hpp>
#include <unordered_map>

template<typename K, typename V>
struct unordered_map_to_list {
    static PyObject* convert(const std::unordered_map<K, V>& m) {
        bp::dict d;
        for (const auto& kv : m){
            d[kv.first] = kv.second;
        }
        return bp::incref(d.ptr());
    }
};

template<typename K, typename V>
void create_map_converter() {
    bp::to_python_converter<std::unordered_map<K, V>, unordered_map_to_list<K, V>>{};
}