#include <boost/make_shared.hpp>

#include "mobilenet_v2.h"

namespace hdnn {

using boost::make_shared;
template <typename Dtype>
MobileNetV2<Dtype>::MobileNetV2(float width_mult) {
    int settings[7][4] = {
        {1, 16,  1, 1},
        {6, 24,  2, 2},
        {6, 32,  3, 2},
        {6, 64,  4, 2},
        {6, 96,  3, 1},
        {6, 160, 3, 2},
        {6, 320, 1, 1},
    };
    int num_stage = 7;
    auto conv1_1 = make_shared<Conv2d<Dtype>>("conv1_1", 3, 32, 3, 2, 1, false);
    auto conv2_1_pw = make_shared<Conv2d<Dtype>>("conv2_1_pw", 32, 32, 1, 1, 0, false);
    auto bn1_1 = make_shared<BatchNorm2d<Dtype>>("bn1_1", 32);
    auto relu = make_shared<ReLU<Dtype>>();

    features = make_shared<Sequential<Dtype>>();
    features->push_back(conv1_1);
    features->push_back(bn1_1);
    features->push_back(relu);

    auto in_channel = int(32 * width_mult);
    auto last_channel = width_mult > 1.0 ? int(1280 * width_mult) : 1280;
    for (int stage = 0; stage < num_stage; stage ++) {
        auto* setting = settings[stage];
        auto out_channel = int(setting[1] * width_mult);
        for (int i = 0; i < setting[2]; i ++) {
            shared_ptr<InvertedResidual> block;
            if (i == 0)
                block = make_shared<InvertedResidual>(in_channel, out_channel, setting[3], setting[0]);
            else
                block = make_shared<InvertedResidual>(in_channel, out_channel, 1, setting[0]);
            features->push_back(block);
            in_channel = out_channel;
        }
    }

    // features->push_back(block2_1);
    // features->push_back(block2_2);
    // for (auto it = block.begin(); it != block.end(); it ++)
    //     features->push_back(*it);
    // features->insert(features.end(), block.begin(), block.end());
}

template <typename Dtype>
Tensor MobileNetV2<Dtype>::operator () (const Tensor& input) {
    Tensor x = input;
    return (*features)(x);
}

template class MobileNetV2<float>;
}
