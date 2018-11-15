#include <boost/make_shared.hpp>

#include "mobilenet_v2.h"
#include "realize.h"

#define MODULE(M, ...) \
    boost::make_shared<hdnn::M<Dtype>>(__VA_ARGS__)

#define REALIZE(M, ...) \
    boost::make_shared<Realize<Dtype>>(boost::make_shared<hdnn::M<Dtype>>(__VA_ARGS__))

namespace hdnn {

using boost::make_shared;
template <typename Dtype>
MobileNetV2<Dtype>::MobileNetV2(float width_mult) {
    int settings[][4] = {
        {1, 16,  1, 1},
        {6, 24,  2, 2},
        {6, 32,  3, 2},
        {6, 64,  4, 1},
        {6, 96,  3, 2},
        {6, 160, 3, 2},
        {6, 320, 1, 1},
    };
    this->num_stages_ = 7;

    auto realize = MODULE (Realize);
    auto conv1_1 = MODULE (Conv2d, "conv1_1", 3, 32, 3, 2, 1, false);
    auto bn1_1   = MODULE (BatchNorm2d, "bn1_1", 32);
    auto relu    = MODULE (ReLU);
    features.push_back(conv1_1);
    features.push_back(bn1_1);
    features.push_back(relu);
    features.push_back(realize);

    auto in_channel = int(32 * width_mult);
    auto last_channel = width_mult > 1.0 ? int(1280 * width_mult) : 1280;
    for (int s = 0; s < num_stages_; s ++) {
        auto* setting = settings[s];
        auto out_channel = int(setting[1] * width_mult);
        for (int i = 0; i < setting[2]; i ++) {
            shared_ptr<Module<Dtype>> block;
            const string block_name = "conv" + std::to_string(s + 2) + "_" + std::to_string(i + 1);
            if (i == 0)
                block = REALIZE(InvertedResidual, block_name, in_channel, out_channel, setting[3], setting[0]);
                // block = make_shared<InvertedResidual>(in_channel, out_channel, setting[3], setting[0]);
            else
                block = REALIZE(InvertedResidual, block_name, in_channel, out_channel, 1, setting[0]);
                // block = make_shared<InvertedResidual<Dtype>>(in_channel, out_channel, 1, setting[0]);
            features.push_back(block);
            in_channel = out_channel;
        }
    }

    features.push_back(make_shared<Conv2d     <Dtype>>("conv1x1", in_channel, last_channel, 1, 1, 0, false));
    features.push_back(make_shared<BatchNorm2d<Dtype>>("conv1x1_bn", last_channel));
    features.push_back(make_shared<ReLU       <Dtype>>());

    classifier.push_back(make_shared<Linear   <Dtype>>("linear", last_channel, 1000));
    classifier.push_back(make_shared<Softmax  <Dtype>>());
    classifier.push_back(realize);
}

template <typename Dtype>
Tensor MobileNetV2<Dtype>::operator () (const Tensor& input) {
    Tensor x = input;
    x = features(x);
    x = pool(x);
    x = classifier(x);
    return x;
}

template <typename Dtype>
Tensor MobileNetV2<Dtype>::pool(const Tensor& x) {
    Func f;
    Var c, n;
    RDom r(0, x.size(0), 0, x.size(1));
    LOG(INFO) << x.size(0) << ", " << x.size(1);
    f(c, n) = Halide::sum(x.func()(r.x, r.y, c, n)) / (x.size(0) * x.size(1));
    f.compute_root();
    return Tensor(f, {x.size(2), x.size(3)});
}

template <typename Dtype>
vector<shared_ptr<Module<Dtype>>> MobileNetV2<Dtype>::flatten() {
    vector<shared_ptr<Module<Dtype>>> m;
    auto flatfeat = features.flatten();
    auto flatclas = classifier.flatten();
    m.insert(m.end(), flatfeat.begin(), flatfeat.end());
    m.insert(m.end(), flatclas.begin(), flatclas.end());
    return m;
}
// template <typename Dtype>
// Halide::Func MobileNetV2<Dtype>::compile_jit(const Tensor& input) {
//     return Func();
// }

template class MobileNetV2<float>;
}
