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
    virtual const string type() const { return "Cifar10Quick"; };
private:
    Sequential<Dtype> features, classifier;
};
}

#endif
