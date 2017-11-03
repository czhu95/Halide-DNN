#include <cmath>
#include "util.h"
#include "vision_layers.h"

namespace hdnn {

template <typename Dtype>
MaxPool2d<Dtype>::MaxPool2d(int kernel_size, int stride, int padding) :
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
}
