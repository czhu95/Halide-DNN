#include <cmath>
#include "util.h"
#include "softmax.h"

namespace hdnn {

template <typename Dtype>
Softmax<Dtype>::Softmax() {}

template <typename Dtype>
Tensor Softmax<Dtype>::operator () (const Tensor& x) {

    Var c, n;
    Func f, exp, scale;

    int softmax_features = x.size()[0];
    exp(c, n) = Halide::exp(x.func()(c, n));
    RDom r(0, softmax_features);
    scale(n) = Halide::sum(exp(r.x, n));
    f(c, n) = exp(c, n) / scale(n);

    return Tensor(f, compute_output_size(x.size()));
}

template <typename Dtype>
vector<int> Softmax<Dtype>::compute_output_size(const vector<int>& input_size) const {
    return vector<int>(input_size);
}

template class Softmax<float>;
}