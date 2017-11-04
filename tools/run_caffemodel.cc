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
#include "cifar10_quick.h"

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "the pretrained weights to initialize finetuning.");

using namespace hdnn;

int main(int argc, char* argv[]) {
    google::SetUsageMessage("commands:\n"
            "  model            caffe prototxt\n"
            "  weights          pretrained weights");
    google::ParseCommandLineFlags(&argc, &argv, true);

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need pretrained model weights.";

    caffe::Net<float> caffe_net(FLAGS_model, caffe::TEST);
    caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

    // Prepare input
    int n = 1;
    int c = 3;
    int h = 32;
    int w = 32;
    int count = n * c * h * w;
    float* input = new float[count];
    caffe::caffe_rng_uniform<float>(count, 0, 1, input);

    // Make a copy for caffe input blob;
    auto input_blob = caffe_net.input_blobs()[0];
    auto output_blob = caffe_net.output_blobs()[0];

    // Run caffe net.
    caffe::caffe_copy(count, input, input_blob->mutable_cpu_data());
    caffe_net.Forward();
    for (int i = 0; i < output_blob->shape(1); i ++)
        LOG(INFO) << output_blob->data_at(0, i, 0, 0);

    Cifar10Quick<float> halide_net;
    halide_net.fromCaffeNet(caffe_net);

    vector<int> input_size {w, h, c, n};
    Buffer<float> input_buffer(input, input_size);
    Tensor halide_input(Func(input_buffer), input_size);

    auto halide_output = halide_net(halide_input);
    Func output_func = halide_output.func();
    output_func.trace_stores();

    Buffer<float> output_buffer = output_func.realize(halide_output.size());

    delete[] input;
    return 0;
}
