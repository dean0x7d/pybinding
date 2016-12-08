#pragma once
#include <initializer_list>

namespace cpb { namespace detail {

/// Prevents unused variable warnings when evaluating variadic parameter packs
template<class... Ts> void eval_unordered(Ts&&...) {}
template<class T> void eval_ordered(std::initializer_list<T>) {}

}} // namespace cpb::detail
