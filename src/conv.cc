#include "util.h"
#include "conv.h"

namespace hdnn {

using boost::shared_ptr;
using caffe::ConvolutionParameter;
using caffe::Blob;

template <typename Dtype>
Conv2d<Dtype>::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool bias) :
    in_channels_(in_channels),
    out_channels_(out_channels),
    kernel_size_(kernel_size),
    stride_(stride),
    pad_(padding),
    bias_term_(bias) {

    // vector<int> weight_shape{out_channels_, in_channels_, kernel_size_, kernel_size_};
    vector<int> weight_size{kernel_size_, kernel_size_, in_channels_, out_channels_};
    weight_ = Buffer<Dtype>(weight_size);

    if (bias_term_) {
        vector<int> bias_size{out_channels_};
        bias_ = Buffer<Dtype>(bias_size);
    }
}

template <typename Dtype>
void Conv2d<Dtype>::CopyParams(vector<shared_ptr<Blob<Dtype>>>& blobs) {
    auto weight_blob = blobs[0];
    CHECK_EQ(weight_blob->shape().size(), 4);
    CHECK_EQ(weight_blob->shape(0), out_channels_);
    CHECK_EQ(weight_blob->shape(1), in_channels_);
    CHECK_EQ(weight_blob->shape(2), kernel_size_);
    CHECK_EQ(weight_blob->shape(3), kernel_size_);
    weight_.copy_from(Buffer<Dtype>(weight_blob->mutable_cpu_data(), reversed(weight_blob->shape())));

    if (bias_term_) {
        auto bias_blob = blobs[1];
        CHECK_EQ(bias_blob->shape().size(), 1);
        CHECK_EQ(bias_blob->shape(0), out_channels_);
        bias_.copy_from(Buffer<Dtype>(bias_blob->mutable_cpu_data(), reversed(bias_blob->shape())));
    }
}

template <typename Dtype>
Tensor Conv2d<Dtype>::operator () (const Tensor& x) {

    Var w, h, c, n;
    Func clamped_x, f;

    clamped_x = Halide::BoundaryConditions::constant_exterior(x.func(), 0.f, x.bounds());
    RDom r(0, kernel_size_, 0, kernel_size_, 0, in_channels_);
    f(w, h, c, n) = Halide::sum(weight_(r.x, r.y, r.z, c) *
            clamped_x(w * stride_ - pad_ + r.x, h * stride_ - pad_ + r.y, r.z, n));
    if (bias_term_)
        f(w, h, c, n) += bias_(c);

    // this->func_(x, y, z, w) += Halide::abs(weight_(x + r.x, y + r.y, z + r.z, w + r.w));
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
