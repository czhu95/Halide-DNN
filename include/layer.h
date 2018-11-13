#ifndef LAYER_H_
#define LAYER_H_
#include "common.h"
#include "caffe/proto/caffe.pb.h"
namespace hdnn {

template <typename Dtype>
class Module : public Module<Dtype> {
public:
    Module(const string& name="") : name_(name) {}
    virtual void copyParams(vector<shared_ptr<Blob<Dtype>>>& blobs) {}
    virtual const string& name() const { return name_; }
    virtual const string type() const = 0;
    virtual bool hasParam() const { return false; };

protected:
    virtual vector<int> compute_output_size(const vector<int>& input_size) const = 0;
    string name_;
};

} //namespace hdnn

#endif
