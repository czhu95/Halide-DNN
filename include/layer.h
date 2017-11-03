#ifndef LAYER_H_
#define LAYER_H_
#include "common.h"
#include "caffe/proto/caffe.pb.h"
namespace hdnn {

using Halide::Func;
using caffe::LayerParameter;

template <typename Dtype>
class Layer {
protected:
    virtual void CopyParams(vector<shared_ptr<Blob<Dtype>>>& blobs) {}
    virtual vector<int> compute_output_size(const vector<int>& input_size) const {};
};

} //namespace hdnn

#endif
