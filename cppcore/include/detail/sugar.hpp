#pragma once
#include <initializer_list>

namespace cpb { namespace detail {

/// Prevents unused variable warnings when evaluating variadic parameter packs
template<class... Ts> void eval_unordered(Ts&&...) {}
template<class T> void eval_ordered(std::initializer_list<T>) {}

template<class It1, class It2>
struct range {
    It1 _begin;
    It2 _end;

    It1 begin() const { return _begin; }
    It2 end() const { return _end; }
};

} // namespace detail

template<class It1, class It2>
detail::range<It1, It2> make_range(It1 begin, It2 end) { return {begin, end}; }

} // namespace cpb
