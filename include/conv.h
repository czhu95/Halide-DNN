#include "caffe/proto/caffe.pb.h"
#include "Halide.h"

using Halide::Buffer;
using caffe::ConvolutionParameter;

template <typename Dtype>
Buffer<Dtype> conv(const ConvolutionParameter& parameter,
        const Buffer<Dtype>* weight, const Buffer<Dtype>* bias);

Buffer<float> conv(const ConvolutionParameter& parameter,
        const Buffer<float>* weight, const Buffer<float>* bias);

