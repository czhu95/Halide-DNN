CXXFLAGS := -std=c++11 -pthread
INCLUDE_DIRS := include caffe/include caffe/build/src Halide/include
LIBRARY_DIRS := /usr/local/lib /usr/lib caffe/build/lib Halide/lib
LIBRARIES := glog gflags boost_system boost_filesystem caffe Halide

# cuda
INCLUDE_DIRS += /usr/local/cuda/include 
LIBRARY_DIRS += /usr/local/cuda/lib64 
LIBRARIES += cudart cublas curand 

# opencv
LIBRARIES += opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs 

FLAGS := $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
LDFLAGS := $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
	$(foreach library,$(LIBRARIES),-l$(library))

# add llvm dependencies
LDFLAGS += $(shell llvm-config --ldflags --system-libs --libs | sed -e 's/\\/\//g' -e 's/\([a-zA-Z]\):/\/\1/g')

all:
	g++ src/main.cc src/conv.cc $(CXXFLAGS) $(FLAGS) $(LDFLAGS) -o main

test_conv: src/test_conv.cc src/conv.cc
	g++ src/test_conv.cc src/conv.cc $(CXXFLAGS) $(FLAGS) $(LDFLAGS) -o test_conv

test_pool: src/test_pool.cc src/pool.cc
	g++ src/test_pool.cc src/pool.cc $(CXXFLAGS) $(FLAGS) $(LDFLAGS) -o test_pool

test_relu: src/test_relu.cc src/relu.cc
	g++ src/test_relu.cc src/relu.cc $(CXXFLAGS) $(FLAGS) $(LDFLAGS) -o test_relu

test_linear: src/test_linear.cc src/linear.cc
	g++ src/test_linear.cc src/linear.cc $(CXXFLAGS) $(FLAGS) $(LDFLAGS) -o test_linear

test_softmax: src/test_softmax.cc src/softmax.cc
	g++ src/test_softmax.cc src/softmax.cc $(CXXFLAGS) $(FLAGS) $(LDFLAGS) -o test_softmax
