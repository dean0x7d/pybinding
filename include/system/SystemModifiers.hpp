#pragma once
#include "support/dense.hpp"
#include <vector>
#include <algorithm>
#include <memory>

namespace tbm {

class SiteStateModifier {
public:
    virtual ~SiteStateModifier() = default;
    
    /// Modify the state (valid or invalid) of sites at the given coordinates
    virtual void apply(ArrayX<bool>& site_state, const CartesianArray& positions) const = 0;
};

class PositionModifier {
public:
    virtual ~PositionModifier() = default;

    /// Modify the positions system sites
    virtual void apply(CartesianArray& positions) const = 0;
};

class SystemModifiers {
public:
    bool add_unique(const std::shared_ptr<const SiteStateModifier>& m);
    bool add_unique(const std::shared_ptr<const PositionModifier>& m);
    void clear();

public:
    // Keep modifiers as unique elements but insertion order must be preserved (don't use std::set)
    std::vector<std::shared_ptr<const SiteStateModifier>> state;
    std::vector<std::shared_ptr<const PositionModifier>> position;
};

} // namespace tbm
