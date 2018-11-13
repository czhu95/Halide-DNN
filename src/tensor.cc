#include "tensor.h"
#include "util.h"

namespace hdnn {

using boost::shared_ptr;
using caffe::ConvolutionParameter;
using caffe::Blob;

Tensor operator+(const Tensor& o1, const Tensor& o2) {
    CHECK(o1.size_ == o2.size_);
    Func f;
    Var x, y, c, n;
    f(x, y, c, n) = o1.func_(x, y, c, n) + o2.func_(x, y, c, n);
    f.compute_root();
    return Tensor(f, o1.size_);
}

Tensor operator-(const Tensor& o1, const Tensor& o2) {
    CHECK(o1.size_ == o2.size_);
    Func f;
    Var x, y, c, n;
    f(x, y, c, n) = o1.func_(x, y, c, n) - o2.func_(x, y, c, n);
    f.compute_root();
    return Tensor(f, o1.size_);
}

Tensor operator*(const Tensor& o1, const Tensor& o2) {
    CHECK(o1.size_ == o2.size_);
    Func f;
    Var x, y, c, n;
    f(x, y, c, n) = o1.func_(x, y, c, n) * o2.func_(x, y, c, n);
    f.compute_root();
    return Tensor(f, o1.size_);
}

Tensor operator/(const Tensor& o1, const Tensor& o2) {
    CHECK(o1.size_ == o2.size_);
    Func f;
    Var x, y, c, n;
    f(x, y, c, n) = o1.func_(x, y, c, n) / o2.func_(x, y, c, n);
    f.compute_root();
    return Tensor(f, o1.size_);
}
} // namespace hdnn
