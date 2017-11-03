#ifndef VISION_LAYERS_H_
#define VISION_LAYERS_H_
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "Halide.h"
#include "layer.h"
#include "tensor.h"

namespace hdnn {

template <typename Dtype>
class Linear : public Layer<Dtype> {
public:
    Linear(int in_features, int out_features, bool bias=true);
    virtual void CopyParams(vector<shared_ptr<Blob<Dtype>>>& blobs);
    virtual Tensor operator () (const Tensor& v);
private:
    virtual vector<int> compute_output_size(const vector<int>& input_size) const;
    int in_features_;
    int out_features_;
    bool bias_term_;
    Buffer<Dtype> weight_;
    Buffer<Dtype> bias_;
};

} // namespace hdnn

#endif