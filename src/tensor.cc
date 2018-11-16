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
    // f.compute_root();
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

// Tensor& Tensor::mean(int dim) {
//     Func f;
//     Var x, y, c, n;
//     RDom(0, size_[dim]) r;
//     switch (dim) {
//         case 0:
//             f(0, y, c, n) = Halide::sum(f(r.x, y, c, n)) / size_[0];
//             break;
//         case 1:
//             f(x, 0, c, n) = Halide::sum(f(x, r.x, c, n)) / size_[1];
//             break;
//         case 2:
//             f(x, y, 0, n) = Halide::sum(f(x, y, r.x, n)) / size_[2];
//             break;
//         case 3:
//             f(x, y, c, 0) = Halide::sum(f(x, y, c, r.x)) / size_[3];
//             break;
//         default:
//             LOG(FATAL) << "dim out of range";
//             break;
//     }
//     auto size = size_;
//     size[dim] = 1;
//     return Tensor(f, size);
// }
} // namespace hdnn
