#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <vector>

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

using caffe::Blob;
using caffe::Layer;
using caffe::Net;
using std::vector;
using boost::shared_ptr;

using Halide::Buffer;
using Halide::RDom;
using Halide::sum;
using Halide::abs;
using Halide::Var;
using Halide::Func;

int main(int argc, char* argv[]) {
    google::SetUsageMessage("commands:\n"
            "  model            caffe prototxt\n"
            "  weights          pretrained weights");
    google::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << FLAGS_model << std::endl;

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition.";
    // CHECK_GT(FLAGS_weights.size(), 0) << "Need pretrained model weights.";

    caffe::Net<float> caffe_net(FLAGS_model, caffe::TEST);
    // caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

    cv::Mat im = cv::imread("data/test.png", cv::IMREAD_COLOR);

    auto& layers = caffe_net.layers();
    std::cout << "Net has " << layers.size() << " layers." << std::endl;
    auto& layer = layers[1];
    auto& blobs = layer->blobs();
    std::cout << "First layer has " << blobs.size() << " parameters." << std::endl;
    const auto parameter = layer->layer_param().convolution_param();

    std::cout << "Weight shape: (";
    for (int i = 0; i < blobs[0]->num_axes(); i ++) {
        std::cout << blobs[0]->shape(i);
        if (i != blobs[0]->num_axes() - 1)
            std::cout << ',';
    }
    std::cout << ')' << std::endl;

    caffe::caffe_rng_uniform<float>(blobs[0]->count(), 0, 1, blobs[0]->mutable_cpu_data());
    std::cout << "Initialized asum (Caffe): " << caffe::caffe_cpu_asum(blobs[0]->count(), blobs[0]->mutable_cpu_data()) << std::endl;

    Buffer<float> weight(blobs[0]->mutable_cpu_data(), blobs[0]->shape());
    Buffer<float> bias(blobs[1]->mutable_cpu_data(), blobs[1]->shape());

    RDom r(weight);

    Var x, y, z, w;
    Func s("s");
    s(x, y, z, w) = (float)0.;
    s(x, y, z, w) += abs(weight(x + r.x, y + r.y, z + r.z, w + r.w));
    Buffer<float> sum = s.realize(1, 1, 1, 1);
    std::cout << "Initialized asum (Halide): " << sum(0, 0, 0, 0) << std::endl;
    return 0;
}
