#include "conv.h"

template <typename Dtype>
Buffer<Dtype> conv(const ConvolutionParameter& parameter,
        const Buffer<Dtype>* weight, const Buffer<Dtype>* bias) {
    int kernel_size = parameter.kernel_size(0);
}

