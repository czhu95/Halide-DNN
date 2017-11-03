#ifndef VISION_LAYERS_H_
#define VISION_LAYERS_H_
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "Halide.h"
#include "layer.h"
#include "tensor.h"

namespace hdnn {

using Halide::Buffer;
using Halide::Func;
using caffe::Blob;
using std::vector;
using boost::shared_ptr;

template <typename Dtype>
class Conv2d : public Layer<Dtype> {
public:
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride=1, int padding=0, bool bias=true);
    virtual void CopyParams(vector<shared_ptr<Blob<Dtype>>>& blobs);
    virtual Tensor operator () (const Tensor& v);
private:
    virtual vector<int> compute_output_size(const vector<int>& input_size) const;
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int pad_;
    bool bias_term_;
    Buffer<Dtype> weight_;
    Buffer<Dtype> bias_;
};

template <typename Dtype>
class MaxPool2d : public Layer<Dtype> {
public:
    MaxPool2d(int kernel_size, int stride=1, int padding=0);
    virtual Tensor operator () (const Tensor& v);
private:
    virtual vector<int> compute_output_size(const vector<int>& input_size) const;
    int kernel_size_;
    int stride_;
    int pad_;
};

} // namespace hdnn

#endif
