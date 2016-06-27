#pragma once
#include <boost/python/data_members.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/return_by_value.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/import.hpp>

#include "numeric/dense.hpp"

namespace boost { namespace python {

/**
 Return a copy
 */
template<class Function>
object return_copy(Function f) {
    return make_function(f, return_value_policy<return_by_value>());
}

template<class Class, class Data>
object return_copy(Data (Class::*pmf)() const) {
    return make_function([pmf](Class& c) -> Data { return (c.*pmf)(); },
                         return_value_policy<return_by_value>());
}

template<class Class, class Data, class = cpp14::enable_if_t<!std::is_function<Data>::value>>
object return_copy(Data Class::* d) {
    return make_getter(d, return_value_policy<return_by_value>());
}

/**
 Return a copy but keep the parent object alive
 */
template<class Function>
object return_internal_copy(Function f) {
    return make_function(f, return_value_policy<
        return_by_value, with_custodian_and_ward_postcall<0, 1>
    >{});
}

template<class Class, class Data,
         class = cpp14::enable_if_t<!std::is_member_function_pointer<Data Class::*>::value>>
object return_internal_copy(Data Class::* d) {
    return make_getter(d, return_value_policy<
        return_by_value, with_custodian_and_ward_postcall<0, 1>
    >{});
}

/**
 Return a reference and keep the parent object alive
 */
template<class Function>
object return_reference(Function f) {
    return make_function(f, return_internal_reference<>{});
}

template<class Class, class Data,
         class = cpp14::enable_if_t<!std::is_member_function_pointer<Data Class::*>::value>>
object return_reference(Data Class::* d) {
    return make_getter(d, return_internal_reference<>{});
}

/**
 Convert to `arrayref` and return like `return_internal_copy`
 */
template<class Class, class Data>
object return_arrayref(Data (Class::*pmf)() const) {
    return make_function(
        [pmf](Class& c) { return cpb::arrayref((c.*pmf)()); },
        return_value_policy<return_by_value, with_custodian_and_ward_postcall<0, 1>>{}
    );
}

template<class Class, class Data, class = cpp14::enable_if_t<!std::is_function<Data>::value>>
object return_arrayref(Data Class::* d) {
    return make_function(
        [d](Class& c) { return cpb::arrayref(c.*d); },
        return_value_policy<return_by_value, with_custodian_and_ward_postcall<0, 1>>{}
    );
}

namespace detail {
    template<class T>
    object with_changed_class(T const& value, char const* class_name, char const* module_name) {
        auto module = import(module_name);
        auto o = object(value);
        o.attr("__class__") = module.attr(class_name);
        return o;
    }
}

template<class Class, class Data>
object extended(Data (Class::*pmf)() const, char const* class_name,
                char const* module_name = "pybinding") {
    return make_function([=](Class const& c) {
        return detail::with_changed_class((c.*pmf)(), class_name, module_name);
    });
}

template<class Class, class Data, class = cpp14::enable_if_t<!std::is_function<Data>::value>>
object extended(Data Class::* d, char const* class_name, char const* module_name = "pybinding") {
    return make_function([=](Class const& c) {
        return detail::with_changed_class(c.*d, class_name, module_name);
    });
}

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

}} // namespace boost::python
