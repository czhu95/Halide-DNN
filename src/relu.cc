#include "relu.h"

namespace hdnn {

template <typename Dtype>
ReLU<Dtype>::ReLU(const string& name) :
    Module<Dtype>(name) {}

template <typename Dtype>
Tensor ReLU<Dtype>::operator () (const Tensor& x) {

    Var w, h, c, n;
    Func f;

    f(w, h, c, n) = Halide::max(Dtype(0.), x.func()(w, h, c, n));
    // f.store_root().compute_root();

    return Tensor(f, compute_output_size(x.size()));
}

template <typename Dtype>
vector<int> ReLU<Dtype>::compute_output_size(const vector<int>& input_size) const {
    return input_size;
}

template class ReLU<float>;
}
