#include "realize.h"

namespace hdnn {

template <typename Dtype>
Realize<Dtype>::Realize(const string& name) :
    Module<Dtype>(name) {}

template <typename Dtype>
Tensor Realize<Dtype>::operator () (const Tensor& x) {

    Func func = x.func();
    Buffer<float> barrier = func.realize(vector<int>(x.size()));
    return Tensor(Func(barrier), x.size());
}

template class Realize<float>;
}
