#pragma once

/// Syntax sugar for resolving overloaded function pointers:
///  - regular: static_cast<Return (Class::*)(Arg0, Arg1, Arg2)>(&Class::func)
///  - sweet:   &Class::func | resolve<Arg0, Arg1, Arg2>()
template<class... Args>
struct resolve {
    template<class Return>
    friend constexpr auto operator|(Return (* pf)(Args...), resolve) noexcept
                                    -> decltype(pf) { return pf; }

    template<class Return, class Class>
    friend constexpr auto operator|(Return (Class::* pmf)(Args...), resolve) noexcept
                                    -> decltype(pmf) { return pmf; }
};

/// Resolve const member function
///  - regular: static_cast<Return (Class::*)(Arg) const>(&Class::func)
///  - sweet:   &Class::func | resolve_const<Arg>()
template<class... Args>
struct resolve_const {
    template<class Return, class Class>
    friend constexpr auto operator|(Return (Class::* pmf)(Args...) const, resolve_const) noexcept
                                    -> decltype(pmf) { return pmf; }
};
