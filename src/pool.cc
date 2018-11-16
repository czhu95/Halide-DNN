#include <cmath>
#include "util.h"
#include "vision_layers.h"

namespace hdnn {

template <typename Dtype>
MaxPool2d<Dtype>::MaxPool2d(const string& name, int kernel_size, int stride, int padding) :
    Module<Dtype>(name),
    kernel_size_(kernel_size),
    stride_(stride),
    pad_(padding) {}

template <typename Dtype>
Tensor MaxPool2d<Dtype>::operator () (const Tensor& x) {

    Var w, h, c, n;
    Func clamped_x, f;

    clamped_x = Halide::BoundaryConditions::repeat_edge(x.func(), x.bounds());
    RDom r(0, kernel_size_, 0, kernel_size_);
    f(w, h, c, n) = Halide::maximum(clamped_x(w * stride_ - pad_ + r.x, h * stride_ - pad_ + r.y, c, n));

    f.compute_root();
    return Tensor(f, compute_output_size(x.size()));
}

template <typename Dtype>
vector<int> MaxPool2d<Dtype>::compute_output_size(const vector<int>& input_size) const {
    vector<int> output_size(input_size);
    for (int i = 0; i < 2; i ++) {
        const int input_dim = input_size[i];
        const int output_dim = std::ceil((input_dim + 2 * pad_ - kernel_size_) / float(stride_)) + 1;
        output_size[i] = output_dim;
    }
    return output_size;
}

template class MaxPool2d<float>;

template <typename Dtype>
AvgPool2d<Dtype>::AvgPool2d(const string& name, int kernel_size, int stride, int padding) :
    Module<Dtype>(name),
    kernel_size_(kernel_size),
    stride_(stride),
    pad_(padding) {}

template <typename Dtype>
Tensor AvgPool2d<Dtype>::operator () (const Tensor& x) {

    Var w, h, c, n;
    Func clamped_x, f;

    clamped_x = Halide::BoundaryConditions::constant_exterior(x.func(), 0.f, x.bounds());
    RDom r(0, kernel_size_, 0, kernel_size_);
    Expr wi = w * stride_ - pad_ + r.x;
    Expr hi = h * stride_ - pad_ + r.y;
    f(w, h, c, n) = Halide::sum(clamped_x(wi, hi, c, n)) /
        Halide::sum(Halide::select(wi < x.size(0) && hi < x.size(1), 1.f, 0.f));

    Var fused;
    f.compute_root();
    f.fuse(c, n, fused);
    f.parallel(fused);
    f.vectorize(w, 16);
    clamped_x.store_root().compute_root();
    return Tensor(f, compute_output_size(x.size()));
}

template <typename Dtype>
vector<int> AvgPool2d<Dtype>::compute_output_size(const vector<int>& input_size) const {
    vector<int> output_size(input_size);
    for (int i = 0; i < 2; i ++) {
        const int input_dim = input_size[i];
        const int output_dim = std::ceil((input_dim + 2 * pad_ - kernel_size_) / float(stride_)) + 1;
        output_size[i] = output_dim;
    }
    return output_size;
}

template class AvgPool2d<float>;
}
