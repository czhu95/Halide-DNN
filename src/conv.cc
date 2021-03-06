#include "util.h"
#include "scheduler.h"
#include "vision_layers.h"

namespace hdnn {

using boost::shared_ptr;
using caffe::ConvolutionParameter;
using caffe::Blob;

template <typename Dtype>
Conv2d<Dtype>::Conv2d(const string& name, int in_channels, int out_channels, int kernel_size, int stride, int padding, bool bias, int groups) :
    Module<Dtype>(name),
    in_channels_(in_channels),
    out_channels_(out_channels),
    kernel_size_(kernel_size),
    stride_(stride),
    pad_(padding),
    groups_(groups),
    bias_term_(bias),
    scheduler_(new DefaultConv2dScheduler<Dtype>()){

    if (groups_ == 1) {
        vector<int> weight_size{kernel_size_, kernel_size_, in_channels_, out_channels_};
        weight_ = Buffer<Dtype>(weight_size);
    } else {
        // support only group = 1 or depthwise separable convs
        CHECK_EQ(groups_, in_channels_);
        CHECK_EQ(in_channels_, out_channels_);
        vector<int> weight_size{kernel_size_, kernel_size_, 1, out_channels_};
        weight_ = Buffer<Dtype>(weight_size);
    }

    if (bias_term_) {
        vector<int> bias_size{out_channels_};
        bias_ = Buffer<Dtype>(bias_size);
    }
}

template <typename Dtype>
void Conv2d<Dtype>::copyParams(vector<shared_ptr<Blob<Dtype>>>& blobs) {
    auto weight_blob = blobs[0];
    CHECK_EQ(weight_blob->shape().size(), 4);
    CHECK_EQ(weight_blob->shape(0), out_channels_);
    if (groups_ == 1)
        CHECK_EQ(weight_blob->shape(1), in_channels_);
    else
        CHECK_EQ(weight_blob->shape(1), 1);
    CHECK_EQ(weight_blob->shape(2), kernel_size_);
    CHECK_EQ(weight_blob->shape(3), kernel_size_);
    weight_.copy_from(Buffer<Dtype>(weight_blob->mutable_cpu_data(), reversed(weight_blob->shape())));

    if (bias_term_) {
        CHECK_EQ(blobs.size(), 2);
        auto bias_blob = blobs[1];
        CHECK_EQ(bias_blob->shape().size(), 1);
        CHECK_EQ(bias_blob->shape(0), out_channels_);
        bias_.copy_from(Buffer<Dtype>(bias_blob->mutable_cpu_data(), reversed(bias_blob->shape())));
    } else {
        CHECK_EQ(blobs.size(), 1);
    }
}

template <typename Dtype>
Tensor Conv2d<Dtype>::operator () (const Tensor& x) {

    // Var w, h, c, n;
    // Func clamped_x, conv, shift;
    Func f;

    pad = Halide::BoundaryConditions::constant_exterior(x.func(), 0.f, x.bounds());
    if (groups_ == 1) {
        RDom r(0, kernel_size_, 0, kernel_size_, 0, in_channels_);
        conv(w, h, c, n) = Halide::sum(weight_(r.x, r.y, r.z, c) *
                pad(w * stride_ - pad_ + r.x, h * stride_ - pad_ + r.y, r.z, n));
    } else {
        RDom r(0, kernel_size_, 0, kernel_size_);
        conv(w, h, c, n) = Halide::sum(weight_(r.x, r.y, 0, c) *
                pad(w * stride_ - pad_ + r.x, h * stride_ - pad_ + r.y, c, n));
    }
    if (bias_term_) {
        shift(w, h, c, n) = conv(w, h, c, n) + bias_(c);
        f = shift;
    } else {
        f = conv;
    }

    scheduler_->plan(this);
    // Var fused, wo, ho, wi, hi;
    // f.store_root().compute_root();
    // f.fuse(c, n, fused);
    // f.parallel(fused);
    // f.tile(w, h, wo, ho, wi, hi, 16, 16);
    // f.vectorize(w, 8);
    // f.vectorize(w, 16);
    // clamped_x.store_root().compute_root();
    // clamped_x.compute_at(f, wi);
    // clamped_x.store_root().compute_at(f, wi);
    return Tensor(f, compute_output_size(x.size()));
}

template <typename Dtype>
vector<int> Conv2d<Dtype>::compute_output_size(const vector<int>& input_size) const {
    vector<int> output_size;
    for (int i = 0; i < 2; i ++) {
        const int input_dim = input_size[i];
        const int output_dim = (input_dim + 2 * pad_ - kernel_size_) / stride_ + 1;
        output_size.push_back(output_dim);
    }
    output_size.push_back(out_channels_);
    output_size.push_back(input_size.back());
    return output_size;
}

template class Conv2d<float>;
}
