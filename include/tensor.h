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

    // TODO: Can we construct a tensor directly from a buffer?

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
    int stride(int n) const {
        int c = 1;
        if (n < 0)
            n += size_.size();
        for (int i = 0; i < n && i < size_.size(); i ++)
            c *= size_[i];
        return c;
    }
    const vector<int>& size() const { return size_; }
    int size(int n) const { return size_[n]; }

    // basic math operations, overriding Halide::Func
    friend Tensor operator+(const Tensor& o1, const Tensor& o2);
    friend Tensor operator-(const Tensor& o1, const Tensor& o2);
    friend Tensor operator*(const Tensor& o1, const Tensor& o2);
    friend Tensor operator/(const Tensor& o1, const Tensor& o2);

private:
    Func func_;
    vector<int> size_;
};

Tensor operator+(const Tensor& o1, const Tensor& o2);
Tensor operator-(const Tensor& o1, const Tensor& o2);
Tensor operator*(const Tensor& o1, const Tensor& o2);
Tensor operator/(const Tensor& o1, const Tensor& o2);

}
#endif
