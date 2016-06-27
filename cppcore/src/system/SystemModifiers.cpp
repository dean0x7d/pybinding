#include "system/SystemModifiers.hpp"
using namespace cpb;

void SystemModifiers::clear() {
    state.clear();
    position.clear();
}

bool SystemModifiers::empty() const {
    return state.empty() && position.empty();
}
