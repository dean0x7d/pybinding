#pragma once
#include "Python.h"
#include <boost/python/data_members.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/return_by_value.hpp>

namespace boost { namespace python {

template<class F>
object internal_ref(F f) {
    return make_function(f, with_custodian_and_ward_postcall<0, 1>{});
}

template <class F>
object const_ref(F f) {
    return make_function(f, return_value_policy<copy_const_reference>());
}

template<class Property>
object by_value(Property pm) {
    return make_getter(pm, return_value_policy<return_by_value>());
}

template<class Property>
object by_const_ref(Property pm) {
    return make_getter(pm, return_value_policy<copy_const_reference>());
}

}}

/**
 Ensure that the current thread is ready to call the Python C API.
 This is required in non-Python created threads in order to acquire the GIL.
 Python code code may only be executed while GILEnsure is in scope.
*/
class GILEnsure {
public:
    GILEnsure() { gstate = PyGILState_Ensure(); }
    ~GILEnsure() { PyGILState_Release(gstate); }

private:
    PyGILState_STATE gstate;
};

template<class Fn>
inline void gil_ensure(Fn lambda) {
    GILEnsure guard;
    lambda();
}

/**
 Release the GIL. Python code must not be called while the GIL is released.
 Has the opposite effect of GILEnsure - it may be called in any thread which
 already holds the GIL in order to allow other Python threads to run.
*/
class GILRelease {
public:
    GILRelease() { save = PyEval_SaveThread(); }
    ~GILRelease() { PyEval_RestoreThread(save); }

private:
    PyThreadState* save;
};

template<class Fn>
inline void gil_release(Fn lambda) {
    GILRelease guard;
    lambda();
}
