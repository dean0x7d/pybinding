#pragma once

#define CPB_EXTERN_TEMPLATE_CLASS(T)            \
  extern template class T<float>;               \
  extern template class T<std::complex<float>>; \
  extern template class T<double>;              \
  extern template class T<std::complex<double>>;

#define CPB_INSTANTIATE_TEMPLATE_CLASS(T)       \
  template class T<float>;                      \
  template class T<std::complex<float>>;        \
  template class T<double>;                     \
  template class T<std::complex<double>>;

#define CPB_EXTERN_TEMPLATE_CLASS_VARGS(T, ...)              \
  extern template class T<float, __VA_ARGS__>;               \
  extern template class T<std::complex<float>, __VA_ARGS__>; \
  extern template class T<double, __VA_ARGS__>;              \
  extern template class T<std::complex<double>, __VA_ARGS__>;

#define CPB_INSTANTIATE_TEMPLATE_CLASS_VARGS(T, ...)         \
  template class T<float, __VA_ARGS__>;                      \
  template class T<std::complex<float>, __VA_ARGS__>;        \
  template class T<double, __VA_ARGS__>;                     \
  template class T<std::complex<double>, __VA_ARGS__>;


#ifndef __has_attribute
# define __has_attribute(x) 0  // Compatibility with non-clang compilers
#endif

#if __has_attribute(always_inline) || defined(__GNUC__)
# define CPB_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
# define CPB_ALWAYS_INLINE __forceinline
#else
# define CPB_ALWAYS_INLINE inline
#endif
