#ifndef COMMON_H_
#define COMMON_H_
#include "Halide.h"
#include "caffe/caffe.hpp"
#include <string>
#include <vector>

namespace hdnn {

using caffe::Caffe;
using caffe::Blob;
using std::vector;
using std::string;
using boost::shared_ptr;

using Halide::Buffer;
using Halide::ImageParam;
using Halide::RDom;
using Halide::Var;
using Halide::Func;
using Halide::Expr;

}

#endif
