#include "util.h"
#include "linear.h"

namespace hdnn {


template <typename Dtype>
Linear<Dtype>::Linear(const string& name, int in_features, int out_features, bool bias) :
    Module<Dtype>(name),
    in_features_(in_features),
    out_features_(out_features),
    bias_term_(bias) {

    vector<int> weight_size{in_features_, out_features_};
    weight_ = Buffer<Dtype>(weight_size);

    if (bias_term_) {
        vector<int> bias_size{out_features_};
        bias_ = Buffer<Dtype>(bias_size);
    }
}

template <typename Dtype>
void Linear<Dtype>::copyParams(vector<shared_ptr<Blob<Dtype>>>& blobs) {
    auto weight_blob = blobs[0];
    // CHECK_EQ(weight_blob->shape().size(), 2);
    CHECK_EQ(weight_blob->shape(0), out_features_);
    CHECK_EQ(weight_blob->count(1), in_features_);
    weight_.copy_from(Buffer<Dtype>(weight_blob->mutable_cpu_data(), {in_features_, out_features_}));

    if (bias_term_) {
        auto bias_blob = blobs[1];
        CHECK_EQ(bias_blob->shape().size(), 1);
        CHECK_EQ(bias_blob->shape(0), out_features_);
        bias_.copy_from(Buffer<Dtype>(bias_blob->mutable_cpu_data(), {out_features_}));
    }
}

template <typename Dtype>
Tensor Linear<Dtype>::operator () (const Tensor& x) {

    Var c, n;
    Func reshape_x, f;

    int input_dim = x.size().size();
    CHECK(input_dim == 2 || input_dim == 4);
    CHECK_EQ(x.stride(-1), in_features_);
    bool need_reshape = input_dim == 4;
    if (need_reshape)
        reshape_x(c, n) = x.func()(
                c % x.stride(1) / x.stride(0),
                c % x.stride(2) / x.stride(1),
                c % x.stride(3) / x.stride(2), n);
    else
        reshape_x = x.func();

    RDom r(0, in_features_);
    f(c, n) = Halide::sum(weight_(r.x, c) * reshape_x(r.x, n));
    if (bias_term_)
        f(c, n) += bias_(c);

    f.compute_root();
    return Tensor(f, compute_output_size(x.size()));
}

template <typename Dtype>
vector<int> Linear<Dtype>::compute_output_size(const vector<int>& input_size) const {
    return {out_features_, input_size.back()};
}

template class Linear<float>;
}
