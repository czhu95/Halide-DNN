#ifndef CONV_H_
#define CONV_H_
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "Halide.h"
#include "layer.h"

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
    virtual Func& operator () (const Func& v);
private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int pad_;
    bool bias_term_;
    Buffer<Dtype> weight_;
    Buffer<Dtype> bias_;
};

} // namespace hdnn

#endif
