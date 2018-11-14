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
#include "mobilenet_v2.h"

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
    caffe::Caffe::set_mode(Caffe::CPU);
    caffe::Net<float> caffe_net("models/mobilenet_v2/test.prototxt", caffe::TEST);
    caffe_net.CopyTrainedLayersFrom("models/mobilenet_v2/mobilenet_v2.caffemodel");
    fLI::FLAGS_minloglevel = 0;

    // Prepare input
    int n = 1;
    int c = 3;
    int h = 224;
    int w = 224;
    int count = n * c * h * w;
    float* input = new float[count];
    caffe::caffe_rng_uniform<float>(count, -1, 1, input);

    // Make a copy for caffe input blob;
    auto input_blob = caffe_net.input_blobs()[0];
    auto output_blob = caffe_net.output_blobs()[0];
    // Run caffe net.
    caffe::caffe_copy(count, input, input_blob->mutable_cpu_data());
    caffe_net.Forward();

    MobileNetV2<float> halide_net;
    halide_net.fromCaffeNet(caffe_net);

    vector<int> input_size {w, h, c, n};
    Buffer<float> input_buffer(input, input_size);
    Tensor halide_input(Func(input_buffer), input_size);

    auto halide_output = halide_net(halide_input);
    Func output_func = halide_output.func();
    // output_func.trace_stores();

    output_func.compile_jit();
    LOG(INFO) << "Compilation done.";
    const auto size = halide_output.size();
    Buffer<float> output_buffer = output_func.realize(size);

    string size_string;
    for (auto it = size.begin(); it != size.end(); it ++) {
        size_string += std::to_string(*it) + " ";
    }
    LOG(INFO) << "Output size (Caffe): " << output_blob->shape_string();
    LOG(INFO) << "Output size (Halide): " << size_string;

    for (int n = 0; n < output_blob->shape(0); n ++)
        for (int c = 0; c < output_blob->shape(1); c ++)
            for (int h = 0; h < output_blob->shape(2); h ++)
                for (int w = 0; w < output_blob->shape(3); w ++)
                    CHECK_LT(std::abs(output_blob->data_at(n, c, h, w) -
                          output_buffer(c, n)), 5e-4) << n << " " << c << " " << h << " " << w << std::endl
                          << output_blob->data_at(n, c, h, w) << ", " << output_buffer(w, h, c, n);

    LOG(INFO) << "Passed.";
    return 0;
}
