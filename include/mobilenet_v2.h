#ifndef MOBILENET_V2_H_
#define MOBILENET_V2_H_
#include "common.h"
#include "net.h"
#include "linear.h"
#include "vision_layers.h"
#include "relu.h"
#include "softmax.h"

namespace hdnn {

template <typename Dtype>
class MobileNetV2 : public Net<Dtype> {
public:
    MobileNetV2();
    virtual Tensor operator () (const Tensor&);
private:
    struct IdenticalInvertedResidual {
        Conv2d<Dtype> conv1_pw, conv2_dw, conv3_pw;
        BatchNorm2d<Dtype> bn;
        InvertedResidual(int in_channels, int out_channels, int stride, float expand_ratio)
            : conv1_pw()

    };
    struct ExpandInvertedResidual
    Conv2d<Dtype> conv1_1;
    Conv2d<Dtype> conv2_1_pw;
    // MaxPool2d<Dtype> max_pool_3x3;
    // AvgPool2d<Dtype> avg_pool_3x3;
    BatchNorm2d<Dtype> bn1_1;
    ReLU<Dtype> relu;
    Softmax<Dtype> softmax;

    vector<Layer<Dtype>*> param_layers_;
};
}

#endif
