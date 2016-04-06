#pragma once

#define TBM_EXTERN_TEMPLATE_CLASS(T)            \
  extern template class T<float>;               \
  extern template class T<std::complex<float>>; \
  extern template class T<double>;              \
  extern template class T<std::complex<double>>;

#define TBM_INSTANTIATE_TEMPLATE_CLASS(T)       \
  template class T<float>;                      \
  template class T<std::complex<float>>;        \
  template class T<double>;                     \
  template class T<std::complex<double>>;


#ifndef __has_attribute
# define __has_attribute(x) 0  // Compatibility with non-clang compilers
#endif

#if __has_attribute(always_inline) || defined(__GNUC__)
# define TBM_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
# define TBM_ALWAYS_INLINE __forceinline
#else
# define TBM_ALWAYS_INLINE inline
#endif
