#ifndef RELU_H_
#define RELU_H_
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "Halide.h"
#include "module.h"
#include "tensor.h"

namespace hdnn {

template <typename Dtype>
class ReLU : public Module<Dtype> {
public:
    ReLU(const string& name);
    ReLU() : ReLU("") {}
    virtual const string type() const { return "ReLU"; }
    virtual Tensor operator () (const Tensor& v);
private:
    virtual vector<int> compute_output_size(const vector<int>& input_size) const;
};

} // namespace hdnn

#endif
