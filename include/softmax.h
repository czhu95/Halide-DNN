#ifndef RELU_H_
#define RELU_H_
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "Halide.h"
#include "layer.h"
#include "tensor.h"

namespace hdnn {

template <typename Dtype>
class Softmax : public Layer<Dtype> {
public:
    Softmax();
    virtual Tensor operator () (const Tensor& v);
private:
    virtual vector<int> compute_output_size(const vector<int>& input_size) const;
};

} // namespace hdnn

#endif
