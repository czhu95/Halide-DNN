#include "util.h"
#include "vision_layers.h"

#include "caffe/util/math_functions.hpp"

#include <cblas.h>

namespace hdnn {

using boost::shared_ptr;
using caffe::ConvolutionParameter;
using caffe::Blob;

template <typename Dtype>
BatchNorm2d<Dtype>::BatchNorm2d(const string& name, int num_channels, float eps, bool affine) :
    Layer<Dtype>(name),
    num_channels_(num_channels),
    eps_(eps),
    affine_(affine) {

    mean_ = Buffer<Dtype>(num_channels_);
    std_ = Buffer<Dtype>(num_channels_);

    if (affine_) {
        scale_ = Buffer<Dtype>(num_channels_);
        shift_ = Buffer<Dtype>(num_channels_);
    }
}

template <typename Dtype>
void BatchNorm2d<Dtype>::copyParams(vector<shared_ptr<Blob<Dtype>>>& blobs) {
    auto mean_blob = blobs[0];
    auto variance_blob = blobs[1];
    CHECK_EQ(mean_blob->shape().size(), 1);
    CHECK_EQ(mean_blob->shape(0), num_channels_);
    CHECK_EQ(variance_blob->shape().size(), 1);
    CHECK_EQ(variance_blob->shape(0), num_channels_);

    mean_.copy_from(Buffer<Dtype>(mean_blob->mutable_cpu_data(), {num_channels_}));
    std_.copy_from(Buffer<Dtype>(variance_blob->mutable_cpu_data(), {num_channels_}));
    const Dtype scale_factor = blobs[2]->cpu_data()[0] == 0 ?
        0 : 1 / blobs[2]->cpu_data()[0];
    caffe::caffe_cpu_scale(num_channels_, scale_factor, mean_.get()->data(), mean_.get()->data());
    caffe::caffe_cpu_scale(num_channels_, scale_factor, std_.get()->data(), std_.get()->data());
    caffe::caffe_add_scalar(num_channels_, eps_, std_.get()->data());
    caffe::caffe_sqrt(num_channels_, std_.get()->data(), std_.get()->data());

    if (affine_) {
        CHECK_EQ(blobs.size(), 5);
        auto scale_blob = blobs[3];
        auto shift_blob = blobs[4];
        CHECK_EQ(scale_blob->shape().size(), 1);
        CHECK_EQ(shift_blob->shape().size(), 1);
        CHECK_EQ(scale_blob->shape(0), num_channels_);
        CHECK_EQ(shift_blob->shape(0), num_channels_);
        scale_.copy_from(Buffer<Dtype>(scale_blob->mutable_cpu_data(), {num_channels_}));
        shift_.copy_from(Buffer<Dtype>(shift_blob->mutable_cpu_data(), {num_channels_}));
    }
}

template <typename Dtype>
Tensor BatchNorm2d<Dtype>::operator () (const Tensor& x) {

    Var w, h, c, n;
    Func f;

    f(w, h, c, n) = (x.func()(w, h, c, n) - mean_(c)) / std_(c);
    if (affine_)
        f(w, h, c, n) = f(w, h, c, n) * scale_(c) + shift_(c);

    f.compute_root();
    return Tensor(f, compute_output_size(x.size()));
}

template class BatchNorm2d<float>;
}
