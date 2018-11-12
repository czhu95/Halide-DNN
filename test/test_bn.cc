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
using caffe::Blob;

using Halide::RDom;
using Halide::sum;
using Halide::abs;

using std::string;
using std::vector;

using namespace hdnn;

int main(int argc, char* argv[]) {
    // disable verbose logging of caffe
    fLI::FLAGS_minloglevel = 2;
    Net<float> caffe_net("models/mobilenet_v2/MobileNet-v2-deploy.prototxt", caffe::TEST);
    fLI::FLAGS_minloglevel = 0;

    vector<shared_ptr<Blob<float>>> params;

    auto& layers = caffe_net.layers();

    int bn_layer_id = 2;
    auto& bn_layer = layers[bn_layer_id];
    auto param = bn_layer->layer_param();
    LOG_IF(INFO, Caffe::root_solver()) << param.type();
    CHECK_EQ(param.type(), "BatchNorm");

    auto& bn_blobs = bn_layer->blobs();
    caffe::caffe_rng_uniform<float>(bn_blobs[0]->count(), 0, 1, bn_blobs[0]->mutable_cpu_data());
    caffe::caffe_rng_uniform<float>(bn_blobs[1]->count(), 0, 1, bn_blobs[1]->mutable_cpu_data());
    caffe::caffe_rng_uniform<float>(bn_blobs[2]->count(), .5, 1, bn_blobs[2]->mutable_cpu_data());
    params.insert(params.end(), bn_blobs.begin(), bn_blobs.end());
    LOG_IF(INFO, Caffe::root_solver()) << "Weight shape (Caffe): " << bn_blobs[0]->shape_string();

    int scale_layer_id = bn_layer_id + 1;
    auto& scale_layer = layers[scale_layer_id];
    param = scale_layer->layer_param();
    LOG_IF(INFO, Caffe::root_solver()) << param.type();
    CHECK_EQ(param.type(), "Scale");

    auto& scale_blobs = scale_layer->blobs();
    caffe::caffe_rng_uniform<float>(scale_blobs[0]->count(), 0, 1, scale_blobs[0]->mutable_cpu_data());
    caffe::caffe_rng_uniform<float>(scale_blobs[1]->count(), 0, 1, scale_blobs[1]->mutable_cpu_data());
    params.insert(params.end(), scale_blobs.begin(), scale_blobs.end());
    LOG_IF(INFO, Caffe::root_solver()) << "Weight shape (Caffe): " << scale_blobs[0]->shape_string();

    auto* input_blob = caffe_net.bottom_vecs()[bn_layer_id][0];
    auto* output_blob = caffe_net.top_vecs()[scale_layer_id][0];
    caffe::caffe_rng_gaussian<float>(input_blob->count(), 0, 1, input_blob->mutable_cpu_data());
    caffe::Blob<float> input_clone;
    // inplace caffe layer overwrites input blob
    input_clone.CopyFrom(*input_blob, false, true);

    caffe_net.ForwardFromTo(bn_layer_id, scale_layer_id);
    LOG_IF(INFO, Caffe::root_solver()) << "bn output shape (Caffe): " << output_blob->shape_string();

    auto channels = input_blob->shape(1);
    auto bn = hdnn::BatchNorm2d<float>(channels);
    bn.copyParams(params);

    Buffer<float> input(input_clone.mutable_cpu_data(), reversed(input_blob->shape()));
    Tensor x(Func(input), reversed(input_blob->shape()));
    x = bn(x);

    Func ppl = x.func();
    Buffer<float> halide_output = ppl.realize(x.size());
    string size_string;
    const auto size = x.size();
    for (auto it = size.begin(); it != size.end(); it ++) {
        size_string += std::to_string(*it) + " ";
    }
    LOG_IF(INFO, Caffe::root_solver()) << "bn output size (Halide): " << size_string;

    for (int n = 0; n < output_blob->shape(0); n ++)
        for (int c = 0; c < output_blob->shape(1); c ++)
            for (int h = 0; h < output_blob->shape(2); h ++)
                for (int w = 0; w < output_blob->shape(3); w ++)
                    CHECK_LT(std::abs(output_blob->data_at(n, c, h, w) - halide_output(w, h, c, n)) , 1e-4) << n << " " << c << " " << h << " " << w;

    LOG_IF(INFO, Caffe::root_solver()) << "Passed.";
    return 0;
}
