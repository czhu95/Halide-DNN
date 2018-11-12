#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <vector>
#include <string>

#include "common.h"
#include "util.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "vision_layers.h"

using caffe::Layer;
using caffe::Net;

using Halide::RDom;
using Halide::sum;
using Halide::abs;

using std::string;

using namespace hdnn;

int main(int argc, char* argv[]) {
    // disable verbose logging of caffe
    fLI::FLAGS_minloglevel = 2;
    caffe::Net<float> caffe_net("models/mobilenet_v2/MobileNet-v2-deploy.prototxt", caffe::TEST);
    fLI::FLAGS_minloglevel = 0;

    int layer_id = 9;
    auto& layers = caffe_net.layers();
    auto& layer = layers[layer_id];
    auto param = layer->layer_param();
    // auto conv_param = param.convolution_param();
    // auto kernel_size = conv_param.kernel_size(0);
    // auto pad = conv_param.pad(0);
    // auto stride = conv_param.stride(0);
    LOG_IF(INFO, Caffe::root_solver()) << param.type();
    CHECK_EQ(param.type(), "Convolution");

    auto& blobs = layer->blobs();
    caffe::caffe_rng_uniform<float>(blobs[0]->count(), 0, 1, blobs[0]->mutable_cpu_data());
    // caffe::caffe_rng_uniform<float>(blobs[1]->count(), 0, 1, blobs[1]->mutable_cpu_data());
    LOG_IF(INFO, Caffe::root_solver()) << "Weight shape (Caffe): " << blobs[0]->shape_string();

    auto* input_blob = caffe_net.bottom_vecs()[layer_id][0];
    auto* output_blob = caffe_net.top_vecs()[layer_id][0];
    caffe::caffe_rng_gaussian<float>(input_blob->count(), 0, 1, input_blob->mutable_cpu_data());

    caffe_net.ForwardFromTo(layer_id, layer_id);
    LOG_IF(INFO, Caffe::root_solver()) << "DepthwiseConv output shape (Caffe): " << output_blob->shape_string();

    auto dwconv_layer = hdnn::DepthwiseConv2d<float>(input_blob->shape(1), 3, 1, 1, false);
    dwconv_layer.copyParams(layer->blobs());

    Buffer<float> input(input_blob->mutable_cpu_data(), reversed(input_blob->shape()));
    Tensor x(Func(input), reversed(input_blob->shape()));
    x = dwconv_layer(x);

    Func ppl = x.func();
    Buffer<float> halide_output = ppl.realize(x.size());
    string size_string;
    const auto size = x.size();
    for (auto it = size.begin(); it != size.end(); it ++) {
        size_string += std::to_string(*it) + " ";
    }
    LOG_IF(INFO, Caffe::root_solver()) << "DepthwiseConv output size (Halide): " << size_string;

    for (int n = 0; n < output_blob->shape(0); n ++)
        for (int c = 0; c < output_blob->shape(1); c ++)
            for (int h = 0; h < output_blob->shape(2); h ++)
                for (int w = 0; w < output_blob->shape(3); w ++)
                    CHECK_LT(std::abs(output_blob->data_at(n, c, h, w) - halide_output(w, h, c, n)) , 1e-5) << n << " " << c << " " << h << " " << w;

    LOG_IF(INFO, Caffe::root_solver()) << "Passed.";
    return 0;
}
