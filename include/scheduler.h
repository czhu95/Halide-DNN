#ifndef SCHEDULE_H_
#define SCHEDULE_H_
#include "Halide.h"
#include "module.h"
#include "vision_layers.h"

namespace hdnn {

using Halide::Func;
using Halide::Var;

// useless parent class for all schedules
// user defined schedules should inherit this class
template<typename Dtype>
class Scheduler {
public:
    virtual void plan(Module<Dtype>* m) const = 0;
};

// template schedule for Conv2d
// users should feel free to define their own schedules.
template<typename Dtype>
class Conv2dScheduler: public Scheduler<Dtype> {
public:
    Conv2dScheduler(){}
protected:
    int nthreads_, tile_, vectorize_;
};

// default schedule for Conv2d
template<typename Dtype>
class DefaultConv2dScheduler: public Scheduler<Dtype> {
public:
    virtual void plan(Module<Dtype>* m) const override {
        auto* op = static_cast<Conv2d<Dtype>*>(m);
        Var fused, wo, ho, wi, hi;
        op->conv.store_root().compute_root();
        op->conv.fuse(op->c, op->n, fused);
        op->conv.vectorize(op->w, 16);
        op->pad.store_root().compute_root();
    }
};

}
#endif
