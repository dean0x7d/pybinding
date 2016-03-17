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
