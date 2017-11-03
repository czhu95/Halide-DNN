#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <vector>

#include "common.h"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "conv.h"

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "the pretrained weights to initialize finetuning.");

using caffe::Layer;
using caffe::Net;

using Halide::RDom;
using Halide::sum;
using Halide::abs;

using namespace hdnn;

int main(int argc, char* argv[]) {
    google::SetUsageMessage("commands:\n"
            "  model            caffe prototxt\n"
            "  weights          pretrained weights");
    google::ParseCommandLineFlags(&argc, &argv, true);

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition.";
    // CHECK_GT(FLAGS_weights.size(), 0) << "Need pretrained model weights.";

    caffe::Net<float> caffe_net(FLAGS_model, caffe::TEST);
    // caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

    cv::Mat im = cv::imread("data/test.png", cv::IMREAD_COLOR);

    auto& layers = caffe_net.layers();
    LOG_IF(INFO, Caffe::root_solver()) << "Net has " << layers.size() << " layers.";
    auto& layer = layers[1];
    auto& blobs = layer->blobs();
    LOG_IF(INFO, Caffe::root_solver()) << "First layer has " << blobs.size() << " parameters.";

    caffe::caffe_rng_uniform<float>(blobs[0]->count(), 0, 1, blobs[0]->mutable_cpu_data());
    LOG_IF(INFO, Caffe::root_solver()) << "Initialized asum (Caffe): " << caffe::caffe_cpu_asum(blobs[0]->count(), blobs[0]->mutable_cpu_data());

    auto conv_layer = hdnn::Conv2d<float>(3, 32, 5);

    conv_layer.CopyParams(layer->blobs());
    caffe::caffe_rng_uniform<float>(blobs[0]->count(), 0, 1, blobs[0]->mutable_cpu_data());
    Func& s = conv_layer(Func());
    Buffer<float> sum = s.realize(1, 1, 1, 1);
    LOG_IF(INFO, Caffe::root_solver()) << "Initialized asum (Halide): " << sum(0, 0, 0, 0);
/*
    const auto parameter = layer->layer_param().convolution_param();

    LOG_IF(INFO, Caffe::root_solver()) << "Weight shape: " << blobs[0]->shape_string();

    caffe::caffe_rng_uniform<float>(blobs[0]->count(), 0, 1, blobs[0]->mutable_cpu_data());
    LOG_IF(INFO, Caffe::root_solver()) << "Initialized asum (Caffe): " << caffe::caffe_cpu_asum(blobs[0]->count(), blobs[0]->mutable_cpu_data());

    Buffer<float> weight(blobs[0]->mutable_cpu_data(), blobs[0]->shape());
    Buffer<float> bias(blobs[1]->mutable_cpu_data(), blobs[1]->shape());

    RDom r(weight);

    Var x, y, z, w;
    Func i("i");
    Func s("s");
    i(x, y, z, w) = weight(x, y, z, w);
    s(x, y, z, w) = (float)0.;
    s(x, y, z, w) += abs(i(x + r.x, y + r.y, z + r.z, w + r.w));
    Buffer<float> sum = s.realize(1, 1, 1, 1);
    LOG_IF(INFO, Caffe::root_solver()) << "Initialized asum (Halide): " << sum(0, 0, 0, 0);
*/
    return 0;
}
