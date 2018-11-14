#ifndef MOBILENET_V2_H_
#define MOBILENET_V2_H_
#include <boost/make_shared.hpp>
#include "common.h"
#include "net.h"
#include "linear.h"
#include "vision_layers.h"
#include "relu.h"
#include "softmax.h"

namespace hdnn {

using boost::make_shared;
template <typename Dtype>
class MobileNetV2 : public Net<Dtype> {
public:
    MobileNetV2(float width_mult=1.0);
    virtual Tensor operator () (const Tensor&);
    virtual const string type() const { return "MobileNetV2"; };
    virtual vector<shared_ptr<Module<Dtype>>> flatten();
private:
    struct InvertedResidual: public Module<Dtype> {
        int stride_;
        bool use_res_connect_;
        shared_ptr<Sequential<Dtype>> conv_;
        InvertedResidual(int in_channels, int out_channels, int stride, float expand_ratio)
            : stride_(stride) {
            use_res_connect_ = stride_ == 1 && in_channels == out_channels;
            auto hidden_dim = int(in_channels * expand_ratio + .5);
            // auto relu = make_shared<ReLU<Dtype>>();
            if (expand_ratio == 0) {
                conv_ = boost::make_shared<Sequential<Dtype>>(vector<shared_ptr<Module<Dtype>>>{
                    make_shared<Conv2d     <Dtype>>(in_channels, hidden_dim, 3, stride, 1, false),
                    make_shared<BatchNorm2d<Dtype>>(hidden_dim),
                    make_shared<ReLU       <Dtype>>(),
                    make_shared<Conv2d     <Dtype>>(hidden_dim, out_channels, 1, 1, 0, false),
                    make_shared<BatchNorm2d<Dtype>>(out_channels),
                });
            } else {
                conv_ = boost::make_shared<Sequential<Dtype>>(vector<shared_ptr<Module<Dtype>>>{
                    make_shared<Conv2d     <Dtype>>(in_channels, hidden_dim, 1, 1, 0, false),
                    make_shared<BatchNorm2d<Dtype>>(hidden_dim),
                    make_shared<ReLU       <Dtype>>(),
                    make_shared<Conv2d     <Dtype>>(hidden_dim, hidden_dim, 3, stride, 1, false, hidden_dim),
                    make_shared<BatchNorm2d<Dtype>>(hidden_dim),
                    make_shared<ReLU       <Dtype>>(),
                    make_shared<Conv2d     <Dtype>>(hidden_dim, out_channels, 1, 1, 0, false),
                    make_shared<BatchNorm2d<Dtype>>(out_channels),
                });
            }
        }

        virtual const string type() const { return "InvertedResisual"; }
        virtual vector<shared_ptr<Module<Dtype>>> modules() { return conv_->modules(); }
        virtual vector<shared_ptr<Module<Dtype>>> flatten() { return conv_->flatten(); }
        virtual Tensor operator () (const Tensor& x) {
            if (use_res_connect_)
                return x + (*conv_)(x);
            else
                return (*conv_)(x);
        }

    };
    // virtual Func compile_jit(const Tensor& input) override;
    Tensor pool(const Tensor& x);
    // vector<shared_ptr<Module<Dtype>>> stages_;
    Sequential<Dtype> features;
    Sequential<Dtype> classifier;
    int num_stages_;
};
}

#endif
