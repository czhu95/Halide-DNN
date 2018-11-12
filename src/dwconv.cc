#include "util.h"
#include "vision_layers.h"

namespace hdnn {

using boost::shared_ptr;
using caffe::ConvolutionParameter;
using caffe::Blob;

template <typename Dtype>
DepthwiseConv2d<Dtype>::DepthwiseConv2d(const string& name, int channels, int kernel_size, int stride, int padding, bool bias) :
    Layer<Dtype>(name),
    channels_(channels),
    kernel_size_(kernel_size),
    stride_(stride),
    pad_(padding),
    bias_term_(bias) {

    vector<int> weight_size{kernel_size_, kernel_size_, 1, channels_};
    weight_ = Buffer<Dtype>(weight_size);

    if (bias_term_) {
        vector<int> bias_size{channels_};
        bias_ = Buffer<Dtype>(bias_size);
    }
}

template <typename Dtype>
void DepthwiseConv2d<Dtype>::copyParams(vector<shared_ptr<Blob<Dtype>>>& blobs) {
    auto weight_blob = blobs[0];
    CHECK_EQ(weight_blob->shape().size(), 4);
    CHECK_EQ(weight_blob->shape(0), channels_);
    CHECK_EQ(weight_blob->shape(1), 1);
    CHECK_EQ(weight_blob->shape(2), kernel_size_);
    CHECK_EQ(weight_blob->shape(3), kernel_size_);
    weight_.copy_from(Buffer<Dtype>(weight_blob->mutable_cpu_data(), reversed(weight_blob->shape())));

    if (bias_term_) {
        auto bias_blob = blobs[1];
        CHECK_EQ(bias_blob->shape().size(), 1);
        CHECK_EQ(bias_blob->shape(0), channels_);
        bias_.copy_from(Buffer<Dtype>(bias_blob->mutable_cpu_data(), reversed(bias_blob->shape())));
    }
}

template <typename Dtype>
Tensor DepthwiseConv2d<Dtype>::operator () (const Tensor& x) {

    Var w, h, c, n;
    Func clamped_x, f;

    clamped_x = Halide::BoundaryConditions::constant_exterior(x.func(), 0.f, x.bounds());
    RDom r(0, kernel_size_, 0, kernel_size_);
    f(w, h, c, n) = Halide::sum(weight_(r.x, r.y, 0, c) *
            clamped_x(w * stride_ - pad_ + r.x, h * stride_ - pad_ + r.y, c, n));
    if (bias_term_)
        f(w, h, c, n) += bias_(c);

    f.compute_root();
    return Tensor(f, compute_output_size(x.size()));
}

template <typename Dtype>
vector<int> DepthwiseConv2d<Dtype>::compute_output_size(const vector<int>& input_size) const {
    vector<int> output_size;
    for (int i = 0; i < 2; i ++) {
        const int input_dim = input_size[i];
        const int output_dim = (input_dim + 2 * pad_ - kernel_size_) / stride_ + 1;
        output_size.push_back(output_dim);
    }
    output_size.push_back(channels_);
    output_size.push_back(input_size.back());
    return output_size;
}

template class DepthwiseConv2d<float>;
}
