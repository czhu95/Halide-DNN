#ifndef CIFAR10_QUICK_H_
#define CIFAR10_QUICK_H_
#include "common.h"
#include "net.h"
#include "linear.h"
#include "vision_layers.h"
#include "relu.h"
#include "softmax.h"

namespace hdnn {

template <typename Dtype>
class Cifar10Quick : public Net<Dtype> {
public:
    Cifar10Quick();
    virtual Tensor operator () (const Tensor&);
private:
    Conv2d<Dtype> conv1, conv2, conv3;
    Linear<Dtype> linear1, linear2;
    MaxPool2d<Dtype> max_pool_3x3;
    AvgPool2d<Dtype> avg_pool_3x3;
    ReLU<Dtype> relu;
    Softmax<Dtype> softmax;
};
}

#endif
