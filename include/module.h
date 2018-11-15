#ifndef MODULE_H_
#define MODULE_H_
#include "common.h"
#include "tensor.h"

namespace hdnn {

using boost::shared_ptr;
template <typename Dtype>
class Module {
public:
    Module() {};
    Module(const string& name) : name_(name) {}
    virtual Tensor operator () (const Tensor& input) = 0;
    virtual vector<shared_ptr<Module<Dtype>>> modules() { return {}; };
    virtual vector<shared_ptr<Module<Dtype>>> flatten() { return {}; };
    virtual void copyParams(vector<shared_ptr<Blob<Dtype>>>& blobs) {}
    virtual const string& name() const { return name_; }
    virtual const string type() const = 0;
    virtual bool hasParam() const { return false; };
protected:
    string name_;
};

template <typename Dtype>
class Sequential : public Module<Dtype> {
public:
    Sequential() {}
    Sequential(const vector<shared_ptr<Module<Dtype>>>& modules) : modules_(modules) {}
    virtual Tensor operator () (const Tensor& input) {
        Tensor x = input;
        for (auto it = modules_.begin(); it != modules_.end(); it ++) {
            x = (*it)->operator()(x);
        }
        return x;
    };
    void push_back(const shared_ptr<Module<Dtype>>& module) { modules_.push_back(module); }
    inline virtual vector<shared_ptr<Module<Dtype>>> modules() { return modules_; }
    virtual vector<shared_ptr<Module<Dtype>>> flatten() {
        vector<shared_ptr<Module<Dtype>>> vec;
        for (auto it = modules_.begin(); it != modules_.end(); it ++) {
            if ((*it)->modules().empty()) {
                vec.push_back(*it);
            } else {
                auto m = (*it)->flatten();
                vec.insert(vec.end(), m.begin(), m.end());
            }
        }
        return vec;
    }
    virtual const string type() const { return "Sequential"; }
protected:
    vector<shared_ptr<Module<Dtype>>> modules_;
};
}

#endif
