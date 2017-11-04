#ifndef NET_H_
#define NET_H_
#include "common.h"
#include "tensor.h"
#include "layer.h"

namespace hdnn {

template <typename Dtype>
class Net {
public:
    Net() {};
    virtual Tensor operator () (const Tensor& input) = 0;

    void fromCaffeNet(caffe::Net<Dtype>& net) {
        auto source_layers = net.layers();
        auto source_layer = source_layers.begin();
        for (auto target_layer = param_layers_.begin();
                target_layer != param_layers_.end();
                target_layer ++) {
            if (!(*target_layer)->hasParams())
                continue;
            const string& target_name = (*target_layer)->name();
            while ((*source_layer)->layer_param().name() != target_name) {
                source_layer ++;
                if (source_layer == source_layers.end())
                    LOG(FATAL) << "Cannot find parameters for " << target_name;
            }
            (*target_layer)->copyParams((*source_layer)->blobs());
            LOG(INFO) << "Copied parameters for " << target_name;
        }
    }

protected:
    vector<Layer<Dtype>*> param_layers_;
};

}
#endif
