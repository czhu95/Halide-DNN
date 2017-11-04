#include "cifar10_quick.h"

namespace hdnn {

template <typename Dtype>
Cifar10Quick<Dtype>::Cifar10Quick() :
    conv1("conv1", 3, 32, 5, 1, 2),
    conv2("conv2", 32, 32, 5, 1, 2),
    conv3("conv3", 32, 64, 5, 1, 2),
    linear1("ip1", 1024, 64),
    linear2("ip2", 64, 10),
    max_pool_3x3(3, 2),
    avg_pool_3x3(3, 2) {

    this->param_layers_.push_back(&conv1);
    this->param_layers_.push_back(&conv2);
    this->param_layers_.push_back(&conv3);
    this->param_layers_.push_back(&linear1);
    this->param_layers_.push_back(&linear2);
}

template <typename Dtype>
Tensor Cifar10Quick<Dtype>::operator () (const Tensor& input) {
    Tensor x = input;
    x = conv1(x);
    x = max_pool_3x3(x);
    x = relu(x);
    x = conv2(x);
    x = relu(x);
    x = avg_pool_3x3(x);
    x = conv3(x);
    x = relu(x);
    x = avg_pool_3x3(x);

    x = linear1(x);
    x = linear2(x);
    x = softmax(x);

    return x;
}

template class Cifar10Quick<float>;
}
