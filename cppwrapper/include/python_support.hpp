#pragma once
#include <boost/python/data_members.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/return_by_value.hpp>

#include "support/dense.hpp"
#include "support/sparseref.hpp"

namespace boost { namespace python {

template<class Function>
object copy_value(Function f) {
    return make_function(f, return_value_policy<return_by_value>());
}

template<class Class, class Data>
object copy_value(Data (Class::*pmf)() const) {
    return make_function([pmf](Class& c) -> Data { return (c.*pmf)(); },
                         return_value_policy<return_by_value>());
}

template<class Class, class Data, class = cpp14::enable_if_t<!std::is_function<Data>::value>>
object copy_value(Data Class::* d) {
    return make_getter(d, return_value_policy<return_by_value>());
}

template<class Function>
object internal_ref(Function f) {
    return make_function(f, return_value_policy<
        return_by_value, with_custodian_and_ward_postcall<0, 1>
    >{});
}

template<class Class, class Data,
         class = cpp14::enable_if_t<!std::is_member_function_pointer<Data Class::*>::value>>
object internal_ref(Data Class::* d) {
    return make_getter(d, return_value_policy<
        return_by_value, with_custodian_and_ward_postcall<0, 1>
    >{});
}

template<class Class, class Data>
object dense_uref(Data (Class::*pmf)() const) {
    return make_function(
        [pmf](Class& c) { return tbm::arrayref((c.*pmf)()); },
        return_value_policy<return_by_value, with_custodian_and_ward_postcall<0, 1>>{}
    );
}

template<class Class, class Data, class = cpp14::enable_if_t<!std::is_function<Data>::value>>
object dense_uref(Data Class::* d) {
    return make_function(
        [d](Class& c) { return tbm::arrayref(c.*d); },
        return_value_policy<return_by_value, with_custodian_and_ward_postcall<0, 1>>{}
    );
}

template<class Class, class Data, class = cpp14::enable_if_t<!std::is_function<Data>::value>>
object sparse_uref(Data Class::* d) {
    return make_function(
        [d](Class& c) { return tbm::SparseURef{c.*d}; },
        return_value_policy<return_by_value, with_custodian_and_ward_postcall<0, 1>>{}
    );
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
