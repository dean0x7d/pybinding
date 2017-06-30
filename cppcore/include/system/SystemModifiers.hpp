#pragma once
#include "Lattice.hpp"

#include "numeric/dense.hpp"

#include <vector>
#include <algorithm>
#include <memory>

namespace cpb {

class Foundation;

/**
 Modify the state (valid or invalid) of lattice sites, e.g. to create vacancies
 */
class SiteStateModifier {
public:
    using Function = std::function<void(Eigen::Ref<ArrayX<bool>> state, CartesianArrayConstRef pos,
                                        string_view sublattice)>;
    Function apply; ///< to be user-implemented
    int min_neighbors; ///< afterwards, remove sites with less than this number of neighbors

    SiteStateModifier(Function const& apply, int min_neighbors = 0)
        : apply(apply), min_neighbors(min_neighbors) {}
};

/**
 Modify the position of lattice sites, e.g. to apply geometric deformations
 */
class PositionModifier {
public:
    using Function = std::function<void(CartesianArrayRef position, string_view sublattice)>;
    Function apply; ///< to be user-implemented

    PositionModifier(Function const& apply) : apply(apply) {}
};

template<class T> void apply(T const&, Foundation&) {}
void apply(SiteStateModifier const& m, Foundation& f);
void apply(PositionModifier const& m, Foundation& f);

/**
 Polymorphic storage for system/foundation modifiers

 Behaves like a common base for several classes but without actually needing
 to inherit from anything -- a class just needs to satisfy the interface.
 This allows us to use value semantics with polymorphic behavior.

 See: "Inheritance Is The Base Class of Evil" by Sean Parent.
 */
class SystemModifier {
public:
    template<class T>
    SystemModifier(T x) : impl(std::make_shared<Storage<T>>(std::move(x))) { }

    friend void apply(SystemModifier const& x, Foundation& f) { x.impl->v_apply(f); }

private:
    struct Interface {
        Interface() = default;
        virtual ~Interface() = default;

        Interface(Interface const&) = delete;
        Interface(Interface&&) = delete;
        Interface& operator=(Interface const&) = delete;
        Interface& operator=(Interface&&) = delete;

        virtual void v_apply(Foundation&) const = 0;
    };

    template<class T>
    struct Storage : Interface {
        Storage(T x) : data(std::move(x)) { }

        void v_apply(Foundation& f) const override { apply(data, f); }

        T data;
    };

    std::shared_ptr<Interface const> impl;
};

} // namespace cpb
