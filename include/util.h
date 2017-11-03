#ifndef UTIL_H_
#define UTIL_H_
#include <algorithm>
#include "common.h"

namespace hdnn {

template <typename T>
T reversed(T v) {
    std::reverse(std::begin(v), std::end(v));
    return v;
}
}

#endif
