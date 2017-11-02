CXXFLAGS := -std=c++11 -pthread
INCLUDE_DIRS := include caffe/include /usr/local/cuda/include caffe/build/src Halide/include k
LIBRARY_DIRS := /usr/local/lib /usr/lib /usr/local/cuda/lib64 caffe/build/lib Halide/lib
LIBRARIES := cudart cublas curand glog gflags boost_system boost_filesystem \
	opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs caffe Halide dl

FLAGS := $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
LDFLAGS := $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
	$(foreach library,$(LIBRARIES),-l$(library))

LDFLAGS += $(shell llvm-config --ldflags --system-libs --libs | sed -e 's/\\/\//g' -e 's/\([a-zA-Z]\):/\/\1/g')

all:
	g++ src/test_caffe.cc $(CXXFLAGS) $(FLAGS) $(LDFLAGS) -o main
	# g++ src/classification.cc $(CXXFLAGS) $(FLAGS) $(LDFLAGS) -o classification
