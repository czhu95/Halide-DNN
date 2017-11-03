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

using std::string;

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
    // caffe::caffe_set<float>(blobs[0]->count(), 1, blobs[0]->mutable_cpu_data());
    caffe::caffe_rng_uniform<float>(blobs[1]->count(), 0, 1, blobs[1]->mutable_cpu_data());
    // caffe::caffe_set<float>(blobs[1]->count(), 0., blobs[1]->mutable_cpu_data());
    LOG_IF(INFO, Caffe::root_solver()) << "Initialized weight asum (Caffe): " << caffe::caffe_cpu_asum(blobs[0]->count(), blobs[0]->mutable_cpu_data());
    LOG_IF(INFO, Caffe::root_solver()) << "Weight shape (Caffe): " << blobs[0]->shape_string();
    string weight_string;
    for (int i = 0; i < blobs[0]->count(); i ++) {
        weight_string += std::to_string(blobs[0]->cpu_data()[i]) + " ";
        if (i % 3 == 2)
            weight_string += "\n";
    }
    // LOG_IF(INFO, Caffe::root_solver()) << "Weight:\n" << weight_string;

    auto* input_blob = caffe_net.input_blobs()[0];
    auto* output_blob = caffe_net.top_vecs()[1][0];
    caffe::caffe_rng_gaussian<float>(input_blob->count(), 0, 1, input_blob->mutable_cpu_data());
    LOG_IF(INFO, Caffe::root_solver()) << "Input asum (Caffe): " << caffe::caffe_cpu_asum(input_blob->count(), input_blob->mutable_cpu_data());

    string input_string;
    for (int i = 0; i < input_blob->count(); i ++) {
        input_string += std::to_string(input_blob->cpu_data()[i]) + " ";
    }
    // LOG_IF(INFO, Caffe::root_solver()) << "Input: " << input_string;

    caffe_net.Forward();
    // LOG_IF(INFO, Caffe::root_solver()) << "Output: " << output_blob->cpu_data()[0] << " " << output_blob->cpu_data()[1];
    LOG_IF(INFO, Caffe::root_solver()) << "Conv output asum (Caffe): " << caffe::caffe_cpu_asum(output_blob->count(), output_blob->mutable_cpu_data());
    LOG_IF(INFO, Caffe::root_solver()) << "Conv output shape (Caffe): " << output_blob->shape_string();

    // auto conv_layer = hdnn::Conv2d<float>(3, 32, 5, 1, 2);
    auto conv_layer = hdnn::Conv2d<float>(input_blob->shape(1), output_blob->shape(1), 5, 1, 2);
    conv_layer.CopyParams(layer->blobs());

    Buffer<float> input(input_blob->mutable_cpu_data(), reversed(input_blob->shape()));
    Tensor x(Func(input), reversed(input_blob->shape()));
    x = conv_layer(x);

    Var n, c, h, w;
    Func ppl = x.func();
    Buffer<float> halide_output = ppl.realize(x.size());
    // Func asum;
    // RDom r(x.bounds());
    // asum(w, h, c, n) = Halide::sum(Halide::abs(x.func()(w + r.x, h + r.y, c + r.z, n + r.w)));
    // Buffer<float> s = asum.realize(1, 1, 1, 1);
    // LOG_IF(INFO, Caffe::root_solver()) << "Conv output asum (Halide): " << s(0, 0, 0, 0);
    string size_string;
    const auto size = x.size();
    for (auto it = size.begin(); it != size.end(); it ++) {
        size_string += std::to_string(*it) + " ";
    }
    LOG_IF(INFO, Caffe::root_solver()) << "Conv output size (Halide): " << size_string;

    for (int n = 0; n < output_blob->shape(0); n ++)
        for (int c = 0; c < output_blob->shape(1); c ++)
            for (int h = 0; h < output_blob->shape(2); h ++)
                for (int w = 0; w < output_blob->shape(3); w ++)
                    CHECK_LT(std::abs(output_blob->data_at(n, c, h, w) - halide_output(w, h, c, n)) , 1e-5) << n << c << h << w;

    LOG_IF(INFO, Caffe::root_solver()) << "Passed.";
    // LOG_IF(INFO, Caffe::root_solver()) << "Output (Halide): " << halide_output(0, 0, 0, 0) << " " << halide_output(0, 0, 1, 0);
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
