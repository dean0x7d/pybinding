#include "system/SystemModifiers.hpp"
using namespace tbm;

void SystemModifiers::clear() {
    state.clear();
    position.clear();
}

bool SystemModifiers::empty() const {
    return state.empty() && position.empty();
}
