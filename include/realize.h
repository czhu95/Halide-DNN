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
    Realize(const string& name, const shared_ptr<Module<Dtype>>& module = {});
    Realize(const shared_ptr<Module<Dtype>>& module = {})
      : Realize("", module) {}
    virtual const string type() const override { return "Realize"; }
    virtual vector<shared_ptr<Module<Dtype>>> modules() override {
        if (module_) return {module_};
        return {};
    }
    virtual vector<shared_ptr<Module<Dtype>>> flatten() override {
        if (module_) return module_->flatten();
        return {};
    }
    virtual Tensor operator () (const Tensor& v);
private:
    virtual vector<int> compute_output_size(const vector<int>& input_size) const {
        return input_size;
    };
    shared_ptr<Module<Dtype>> module_;
};

} // namespace hdnn

#endif
