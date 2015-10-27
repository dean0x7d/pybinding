#pragma once
#include "support/dense.hpp"
#include <vector>
#include <algorithm>
#include <memory>

namespace tbm {

class SiteStateModifierImpl {
public:
    virtual ~SiteStateModifierImpl() = default;
    
    /// Modify the state (valid or invalid) of sites at the given coordinates
    virtual void apply(ArrayX<bool>& site_state, const CartesianArray& positions) const = 0;
};

class PositionModifierImpl {
public:
    virtual ~PositionModifierImpl() = default;

    /// Modify the positions system sites
    virtual void apply(CartesianArray& positions) const = 0;
};

using SiteStateModifier = std::shared_ptr<SiteStateModifierImpl const>;
using PositionModifier = std::shared_ptr<PositionModifierImpl const>;

class SystemModifiers {
public:
    bool add_unique(SiteStateModifier const& m);
    bool add_unique(PositionModifier const& m);
    void clear();

public:
    // Keep modifiers as unique elements but insertion order must be preserved (don't use std::set)
    std::vector<SiteStateModifier> state;
    std::vector<PositionModifier> position;
};

} // namespace tbm
