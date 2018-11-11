CXX := g++
BUILD_DIR := build

CXXFLAGS := -std=c++11 -pthread
INCLUDE_DIRS := include caffe/build/install/include caffe/build/src Halide/include
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

SRCS := $(shell find src -name "*.cc" -printf "%f\n")
TOOL_SRCS := $(shell find tools -name "*.cc")
TEST_SRCS := $(shell find test -name "*.cc")

OBJS := $(addprefix $(BUILD_DIR)/, ${SRCS:.cc=.o})

TOOL_BINS := $(addprefix $(BUILD_DIR)/, ${TOOL_SRCS:.cc=})
TEST_BINS := $(addprefix $(BUILD_DIR)/, ${TEST_SRCS:.cc=})

all: $(TEST_BINS) $(TOOL_BINS)

$(TEST_BINS): $(BUILD_DIR)/% : %.cc $(OBJS)
	@ mkdir -p $(BUILD_DIR)/test
	@ echo LD $@
	@ $(CXX) $^ -o $@ $(CXXFLAGS) $(FLAGS) $(LDFLAGS)

$(TOOL_BINS): $(BUILD_DIR)/% : %.cc $(OBJS)
	@ mkdir -p $(BUILD_DIR)/tools
	@ echo LD $@
	@ $(CXX) $^ -o $@ $(CXXFLAGS) $(FLAGS) $(LDFLAGS)

$(OBJS): $(BUILD_DIR)/%.o : src/%.cc include/*.h
	@ mkdir -p $(BUILD_DIR)
	@ echo CXX -o $@
	@ $(CXX) -c $< -o $@ $(CXXFLAGS) $(FLAGS) $(LDFLAGS)

clean:
	@- rm -rf build
