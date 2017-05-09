#pragma once
#ifdef _MSC_VER // suppress 'static_visitor' deprecation warning
# pragma warning(disable : 4996)
#endif
#include <mapbox/variant.hpp>

namespace cpb { namespace var {

using namespace mapbox::util;

/// Variant of a container with real elements
template<template<class> class... C>
using real = var::variant<C<float>..., C<double>...>;

/// Variant of a container with real or complex elements
template<template<class> class... C>
using complex = var::variant<C<float>..., C<std::complex<float>>...,
                             C<double>..., C<std::complex<double>>...>;

template<class T> struct tag {};

using scalar_tag = var::complex<tag>;

}} // namespace cpb::var
