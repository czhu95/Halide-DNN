#ifndef TENSOR_H_
#define TENSOR_H_
#include <utility>
#include "common.h"

namespace hdnn {
using std::pair;

class Tensor {
public:
    Tensor(const Func& func, const vector<int>& size) :
        func_(func), size_(size) {}
    Tensor(const Func& func, const vector<int>&& size) :
        func_(func), size_(size) {}
    const Func& func() const { return func_; }
    const vector<pair<Expr, Expr>> bounds() const {
        vector<pair<Expr, Expr>> bounds;
        for (auto it = size_.begin(); it != size_.end(); it ++)
            bounds.push_back({0, *it});
        return bounds;
    }
    const vector<pair<Expr, Expr>> bounds(const vector<int>& dims) const {
        vector<pair<Expr, Expr>> bounds;
        for (auto it = dims.begin(); it != dims.end(); it ++)
            bounds.push_back({0, size_[*it]});
        return bounds;
    }
    const vector<int>& size() const { return size_; }
private:
    Func func_;
    vector<int> size_;
};
}
#endif
