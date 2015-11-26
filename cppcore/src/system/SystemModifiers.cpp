#include "system/SystemModifiers.hpp"
using namespace tbm;

bool SystemModifiers::add_unique(SiteStateModifier const& m) {
    if (std::find(state.begin(), state.end(), m) == state.end()) {
        state.push_back(m);
        return true;
    }
    return false;
}

bool SystemModifiers::add_unique(PositionModifier const& m) {
    if (std::find(position.begin(), position.end(), m) == position.end()) {
        position.push_back(m);
        return true;
    }
    return false;
}

void SystemModifiers::clear() {
    state.clear();
    position.clear();
}

bool SystemModifiers::empty() const {
    return state.empty() && position.empty();
}
