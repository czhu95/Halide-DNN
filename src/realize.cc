#include "realize.h"

namespace hdnn {

template <typename Dtype>
Realize<Dtype>::Realize(const string& name) :
    Module<Dtype>(name) {}

template <typename Dtype>
Tensor Realize<Dtype>::operator () (const Tensor& x) {
    t_ = x;
    t_.func().compile_jit();
    barrier_ = ImageParam(Halide::type_of<Dtype>(), t_.size().size());
    return Tensor(barrier_, x.size());
}

template <typename Dtype>
Buffer<Dtype> Realize<Dtype>::run(Buffer<Dtype>& input) {
    Buffer<float> barrier = t_.func().realize(vector<int>(t_.size()));
    barrier_.set(barrier);
    return barrier;
}

template class Realize<float>;
}
