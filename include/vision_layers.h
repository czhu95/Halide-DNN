#ifndef VISION_LAYERS_H_
#define VISION_LAYERS_H_
#include "caffe/proto/caffe.pb.h"
#include "Halide.h"
#include "module.h"
#include "tensor.h"

namespace hdnn {

using Halide::Buffer;
using Halide::Func;
using caffe::Blob;
using std::vector;
using boost::shared_ptr;

// forward declaration
template <typename Dtype> class Scheduler;

template <typename Dtype>
class Conv2d : public Module<Dtype> {
public:
    Conv2d(const string& name, int in_channels, int out_channels, int kernel_size, int stride=1, int padding=0, bool bias=true, int groups=1);
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride=1, int padding=0, bool bias=true, int groups=1)
        : Conv2d("", in_channels, out_channels, kernel_size, stride, padding, bias, groups) {}
    virtual const string type() const { return "Conv2d"; }
    virtual void copyParams(vector<shared_ptr<Blob<Dtype>>>& blobs);
    virtual bool hasParam() const { return true; }
    virtual Tensor operator () (const Tensor& v);
    // expose algorithm details for scheduling
    Var w, h, c, n;
    Func pad, conv, shift;
private:
    virtual vector<int> compute_output_size(const vector<int>& input_size) const;
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int pad_;
    int groups_;
    bool bias_term_;
    Buffer<Dtype> weight_;
    Buffer<Dtype> bias_;
    shared_ptr<Scheduler<Dtype>> scheduler_;
};


template <typename Dtype>
class BatchNorm2d : public Module<Dtype> {
public:
    BatchNorm2d(const string& name, int num_channels, float eps=1e-5, bool affine=true);
    BatchNorm2d(int num_channels, float eps=1e-5, bool affine=true)
        : BatchNorm2d("", num_channels, eps, affine) {}
    virtual const string type() const { return "BatchNorm2d"; }
    virtual void copyParams(vector<shared_ptr<Blob<Dtype>>>& blobs);
    virtual bool hasParam() const { return true; }
    virtual Tensor operator () (const Tensor& v);

private:
    virtual vector<int> compute_output_size(const vector<int>& input_size) const {
        return input_size;
    }

    int num_channels_;
    bool affine_;
    float eps_;
    Buffer<Dtype> mean_, std_, scale_, shift_;
};


template <typename Dtype>
class MaxPool2d : public Module<Dtype> {
public:
    MaxPool2d(const string& name, int kernel_size, int stride=1, int padding=0);
    MaxPool2d(int kernel_size, int stride=1, int padding=0)
        : MaxPool2d("", kernel_size, stride, padding) {}
    virtual const string type() const { return "MaxPool2d"; }
    virtual Tensor operator () (const Tensor& v);
private:
    virtual vector<int> compute_output_size(const vector<int>& input_size) const;
    int kernel_size_;
    int stride_;
    int pad_;
};

template <typename Dtype>
class AvgPool2d : public Module<Dtype> {
public:
    AvgPool2d(const string& name, int kernel_size=0, int stride=1, int padding=0);
    AvgPool2d(int kernel_size=0, int stride=1, int padding=0)
        : AvgPool2d("", kernel_size, stride, padding) {}
    virtual const string type() const { return "AvgPool2d"; }
    virtual Tensor operator () (const Tensor& v);
private:
    virtual vector<int> compute_output_size(const vector<int>& input_size) const;
    int kernel_size_;
    int stride_;
    int pad_;
};

} // namespace hdnn

#endif
