#ifndef NET_H_
#define NET_H_
#include "common.h"
#include "tensor.h"
#include "module.h"
#include "realize.h"

namespace hdnn {

template <typename Dtype>
class Net {
public:
    Net() {};
    // virtual const string type() const { return "Net"; };
    virtual void fromCaffeNet(caffe::Net<Dtype>& net) {
        auto source_layers = net.layers();
        auto source_layer = source_layers.begin();
        auto param_layers = this->flatten();
        for (auto target_layer = param_layers.begin();
                target_layer != param_layers.end();
                target_layer ++) {
            if (!(*target_layer)->hasParam())
                continue;
            const string& target_name = (*target_layer)->name();
            const string& target_type = (*target_layer)->type();
            while (source_layer != source_layers.end()
                   && !matchCaffeType(target_type, (*source_layer)->layer_param().type()))
                source_layer ++;
            CHECK(source_layer != source_layers.end()) << "Cannot find parameters for " << target_name;
            if (target_type == "BatchNorm2d"
                    && source_layer + 1 != source_layers.end()
                    && (*(source_layer + 1))->layer_param().type() == "Scale") {
                // blob array to hold blobs for bn and scale layers
                vector<shared_ptr<Blob<Dtype>>> param_buffer;
                // insert bn blobs
                const string& bn_name = (*source_layer)->layer_param().name();
                auto& bn_blobs = (*source_layer++)->blobs();
                param_buffer.insert(param_buffer.end(),
                        bn_blobs.begin(), bn_blobs.end());
                // insert scale blobs
                const string& scale_name = (*source_layer)->layer_param().name();
                auto& scale_blobs = (*source_layer++)->blobs();
                param_buffer.insert(param_buffer.end(),
                        scale_blobs.begin(), scale_blobs.end());
                // copy from blob buffer array
                LOG(INFO) << "Copying parameters for " << target_name
                    << " (" << bn_name << ", " << scale_name << ")";
                (*target_layer)->copyParams(param_buffer);
            } else {
                const string& source_name = (*source_layer)->layer_param().name();
                LOG(INFO) << "Copying parameters for " << target_name << " (" << source_name << ")";
                (*target_layer)->copyParams((*source_layer++)->blobs());
            }
        }
    }
    virtual Tensor operator () (const Tensor& input) = 0;
    virtual Buffer<Dtype> run(Buffer<Dtype>& input) {
        Buffer<Dtype> out = input;
        for (auto it = realizations_.begin(); it != realizations_.end(); it ++)
            out = (*it)->run(out);
        return out;
    }
    virtual vector<shared_ptr<Module<Dtype>>> flatten() = 0;
    // virtual Func compile_jit(const Tensor& input) { return Halide::Func(); }

protected:
    void collect_realizations() {
        realizations_.clear();
        auto flat = flatten();
        for (auto it = flat.begin(); it != flat.end(); it ++) {
            if ((*it)->type() == "Realize")
                realizations_.push_back(
                        boost::dynamic_pointer_cast<Realize<Dtype>>((*it)));
        }
    }
    bool matchCaffeType(const string& hdnn_type, const string& caffe_type) {
        if (hdnn_type == "Conv2d")
            return caffe_type == "Convolution";
        else if (hdnn_type == "BatchNorm2d")
            return caffe_type == "BatchNorm";
        else if (hdnn_type == "Linear")
            // Allow loading 1x1 conv as linear module
            return caffe_type == "InnerProduct" || caffe_type == "Convolution";
        else if (hdnn_type == "Pooling")
            return caffe_type == "Pooling";
        else
            return caffe_type == hdnn_type;
    }
    vector<shared_ptr<Realize<Dtype>>> realizations_;
    ImageParam input_;
};

}
#endif
