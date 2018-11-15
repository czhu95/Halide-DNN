#include "realize.h"

namespace hdnn {

template <typename Dtype>
Realize<Dtype>::Realize(const string& name, const shared_ptr<Module<Dtype>>& module) :
    Module<Dtype>(name), module_(module) {}

template <typename Dtype>
Tensor Realize<Dtype>::operator () (const Tensor& x) {
    auto out = x;
    if (module_)
      out = (*module_)(x);
    Func func = out.func();
    Buffer<float> barrier = func.realize(vector<int>(out.size()));
    return Tensor(Func(barrier), out.size());
}

template class Realize<float>;
}
