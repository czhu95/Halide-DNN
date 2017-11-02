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

    vector<int> weight_shape{out_channels_, in_channels_, kernel_size_, kernel_size_};
    for (auto it = weight_shape.begin(); it != weight_shape.end(); it ++)
        LOG_IF(INFO, Caffe::root_solver()) << *it;
    weight_ = Buffer<Dtype>(weight_shape);

    if (bias_term_) {
        vector<int> bias_shape{out_channels_};
        bias_ = Buffer<Dtype>(bias_shape);
    }
}

template <typename Dtype>
void Conv2d<Dtype>::CopyParams(vector<shared_ptr<Blob<Dtype>>>& blobs) {
    auto weight_blob = blobs[0];
    weight_.copy_from(Buffer<Dtype>(weight_blob->mutable_cpu_data(), weight_blob->shape()));
    if (bias_term_) {
        auto bias_blob = blobs[1];
        bias_.copy_from(Buffer<Dtype>(bias_blob->mutable_cpu_data(), bias_blob->shape()));
    }
}

template <typename Dtype>
Func& Conv2d<Dtype>::operator () (const Func& v) {

    RDom r(weight_);
    Var x, y, z, w;
    this->func_(x, y, z, w) = (float)0.;
    this->func_(x, y, z, w) += Halide::abs(weight_(x + r.x, y + r.y, z + r.z, w + r.w));
    return this->func_;
}

template class Conv2d<float>;
}
