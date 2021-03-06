#ifndef REALIZE_H_
#define REALIZE_H_
#include <vector>
#include "Halide.h"
#include "module.h"
#include "tensor.h"

namespace hdnn {

template <typename Dtype>
class Realize : public Module<Dtype> {
public:
    Realize(const string& name);
    Realize() : Realize("") {}
    virtual const string type() const override { return "Realize"; }
    virtual Tensor operator () (const Tensor& v);
    virtual Buffer<Dtype> run(Buffer<Dtype>& input);
private:
    virtual vector<int> compute_output_size(const vector<int>& input_size) const {
        return input_size;
    };
    ImageParam barrier_;
    Tensor t_;
};

} // namespace hdnn

#endif
