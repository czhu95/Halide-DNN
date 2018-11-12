#ifndef NET_H_
#define NET_H_
#include "common.h"
#include "tensor.h"
#include "layer.h"

namespace hdnn {

template <typename Dtype>
class Net : public Module<Dtype> {
public:
    Net() {};
    void fromCaffeNet(caffe::Net<Dtype>& net) {
        auto source_layers = net.layers();
        auto source_layer = source_layers.begin();
        for (auto target_layer = param_layers_.begin();
                target_layer != param_layers_.end();
                target_layer ++) {
            if (!(*target_layer)->hasParams())
                continue;
            const string& target_name = (*target_layer)->name();
            const string& target_type = (*target_layer)->type();
            while (!matchCaffeType(target_type,
                        (*source_layer)->layer_param().type())) {
                if (++source_layer == source_layers.end())
                    LOG(FATAL) << "Cannot find parameters for " << target_name;
            }
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
                (*target_layer)->copyParams(param_buffer);
                LOG(INFO) << "Copied parameters for " << target_name
                    << " (" << bn_name << ", " << scale_name << ")";
            } else {
                const string& source_name = (*source_layer)->layer_param().name();
                (*target_layer)->copyParams((*source_layer++)->blobs());
                LOG(INFO) << "Copied parameters for " << target_name << " (" << source_name << ")";
            }
        }
    }

protected:
    bool matchCaffeType(const string& hdnn_type, const string& caffe_type) {
        if (hdnn_type == "Conv2d")
            return caffe_type == "Convolution";
        else if (hdnn_type == "BatchNorm2d")
            return caffe_type == "BatchNorm";
        else if (hdnn_type == "Linear")
            return caffe_type == "InnerProduct";
        else if (hdnn_type == "Pooling")
            return caffe_type == "Pooling";
        else
            return caffe_type == hdnn_type;
    }
};

}
#endif
