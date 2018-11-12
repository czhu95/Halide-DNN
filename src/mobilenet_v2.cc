#include "mobilenet_v2.h"

namespace hdnn {

template <typename Dtype>
MobileNetV2<Dtype>::MobileNetV2() :
      conv1_1("conv1_1", 3, 32, 3, 2, 1, false)
    , conv2_1_pw("conv2_1_pw", 32, 32, 1, 1, 0, false)
    // , conv2("conv2", 32, 32, 5, 1, 2)
    // , conv3("conv3", 32, 64, 5, 1, 2)
    , bn1_1("bn1_1", 32)
    // , linear1("ip1", 1024, 64)
    // , linear2("ip2", 64, 10)
    // , max_pool_3x3(3, 2)
    // , avg_pool_3x3(3, 2)
{
    this->param_layers_.push_back(&conv1_1);
    this->param_layers_.push_back(&bn1_1);
    this->param_layers_.push_back(&conv2_1_pw);
    // this->param_layers_.push_back(&conv2);
    // this->param_layers_.push_back(&conv3);
    // this->param_layers_.push_back(&linear1);
    // this->param_layers_.push_back(&linear2);
}

template <typename Dtype>
Tensor MobileNetV2<Dtype>::operator () (const Tensor& input) {
    Tensor x = input;
    x = conv1_1(x);
    x = bn1_1(x);
    x = relu(x);
    x = conv2_1_pw(x);
    return x;
}

template class MobileNetV2<float>;
}
