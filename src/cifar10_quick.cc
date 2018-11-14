#include <boost/make_shared.hpp>

#include "cifar10_quick.h"

namespace hdnn {

using boost::make_shared;

template <typename Dtype>
Cifar10Quick<Dtype>::Cifar10Quick() {

    auto max_pool = make_shared<MaxPool2d<Dtype>>(3, 2);
    auto avg_pool = make_shared<AvgPool2d<Dtype>>(3, 2);
    auto relu = make_shared<ReLU<Dtype>>();
    features.push_back(make_shared<Conv2d<Dtype>>("conv1", 3, 32, 5, 1, 2));
    features.push_back(max_pool);
    features.push_back(relu);
    features.push_back(make_shared<Conv2d<Dtype>>("conv2", 32, 32, 5, 1, 2));
    features.push_back(relu);
    features.push_back(avg_pool);
    features.push_back(make_shared<Conv2d<Dtype>>("conv3", 32, 64, 5, 1, 2));
    features.push_back(relu);
    features.push_back(avg_pool);

    classifier.push_back(make_shared<Linear<Dtype>>("ip1", 1024, 64));
    classifier.push_back(make_shared<Linear<Dtype>>("ip1", 64, 10));
    classifier.push_back(make_shared<Softmax<Dtype>>());
}

template <typename Dtype>
Tensor Cifar10Quick<Dtype>::operator () (const Tensor& input) {
    Tensor x = input;
    x = features(x);
    x = classifier(x);
    return x;
}


template <typename Dtype>
vector<shared_ptr<Module<Dtype>>> Cifar10Quick<Dtype>::flatten() {
    vector<shared_ptr<Module<Dtype>>> m;
    auto flatfeat = features.flatten();
    auto flatclas = classifier.flatten();
    m.insert(m.end(), flatfeat.begin(), flatfeat.end());
    m.insert(m.end(), flatclas.begin(), flatclas.end());
    return m;
}

template class Cifar10Quick<float>;
}
