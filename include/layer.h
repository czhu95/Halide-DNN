#ifndef LAYER_H_
#define LAYER_H_
#include "common.h"
#include "caffe/proto/caffe.pb.h"
namespace hdnn {

using Halide::Func;
using caffe::LayerParameter;

template <typename Dtype>
class Layer {
public:
    // Layer(const LayerParameter& param) :
        // param_(param) {}
protected:
    virtual void CopyParams(vector<shared_ptr<Blob<Dtype>>>& blobs) {}
    // LayerParameter param_;
    // Halide function for this layer. I think we need it stored somewhere.
    Func func_;
};

} //namespace hdnn

#endif
