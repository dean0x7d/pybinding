#pragma once
#include <type_traits>

namespace tbm {

template<class...> struct TypeList {};

namespace tl {

template<class List1, class List2>
struct ConcatImpl;

template<class List1, class List2>
using Concat = typename ConcatImpl<List1, List2>::type;

template<class... Ts1, class... Ts2>
struct ConcatImpl<TypeList<Ts1...>, TypeList<Ts2...>> {
    using type = TypeList<Ts1..., Ts2...>;
};


template<class T1, class T2>
struct CombinationsImpl;

template<class T1, class T2>
using Combinations = typename CombinationsImpl<T1, T2>::type;

template<class T1, class... Ts2>
struct CombinationsImpl<TypeList<T1>, TypeList<Ts2...>> {
    using type = TypeList<TypeList<T1, Ts2>...>;
};

template<class T1, class... Tail, class... Ts2>
struct CombinationsImpl<TypeList<T1, Tail...>, TypeList<Ts2...>> {
    using type = Concat<
        TypeList<TypeList<T1, Ts2>...>,
        Combinations<
            TypeList<Tail...>, TypeList<Ts2...>
        >
    >;
};


template<class List, template<class> class Predicate>
struct FilterImpl;

template<class List, template<class> class Predicate>
using Filter = typename FilterImpl<List, Predicate>::type;

template<template<class> class Predicate>
struct FilterImpl<TypeList<>, Predicate> {
    using type = TypeList<>;
};

template<class T, class... Ts, template<class> class Predicate>
struct FilterImpl<TypeList<T, Ts...>, Predicate> {
    using type = Concat<
        typename std::conditional<Predicate<T>::value, TypeList<T>, TypeList<>>::type,
        Filter<TypeList<Ts...>, Predicate>
    >;
};

}} // namespace tbm:tl
