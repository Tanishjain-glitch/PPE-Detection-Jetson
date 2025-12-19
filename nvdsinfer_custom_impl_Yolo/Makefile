TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

CC := g++
NVCC := /usr/local/cuda/bin/nvcc

# Common compiler flags
CFLAGS := -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS += -I/opt/nvidia/deepstream/deepstream/sources/includes
CFLAGS += -I/usr/local/cuda/include

# CUDA compiler flags
CUFLAGS := -std=c++14 -shared -Xcompiler -fPIC
CUFLAGS += -I/opt/nvidia/deepstream/deepstream/sources/includes
CUFLAGS += -I/usr/local/cuda/include

# Libraries (common)
LIBS := -lnvinfer_plugin -lnvinfer -L/usr/local/cuda/lib64 -lcudart -lcublas

# Linker flags for CPU (g++)
LFLAGS_CPU := -shared -Wl,--start-group $(LIBS) -Wl,--end-group

# Linker flags for CUDA (nvcc doesn't need -Wl)
LFLAGS_CUDA := -shared $(LIBS)

# Gencode for different architectures
ifeq ($(TARGET_DEVICE),aarch64)
    # Jetson architectures
    GENCODE_FLAGS := -gencode arch=compute_53,code=sm_53   # Jetson Nano
    GENCODE_FLAGS += -gencode arch=compute_62,code=sm_62  # Jetson TX2
    GENCODE_FLAGS += -gencode arch=compute_72,code=sm_72  # Jetson Xavier/NX
    GENCODE_FLAGS += -gencode arch=compute_87,code=sm_87  # Jetson Orin
else
    # Desktop GPUs
    SMS ?= 75 80 86 89 90
    GENCODE_FLAGS := $(foreach sm,$(SMS),-gencode arch=compute_$(sm),code=sm_$(sm))
endif

# Source files
CPU_SRC := nvdsparsebbox_Yolo.cpp
CUDA_SRC := nvdsparsebbox_Yolo_cuda.cu

# Target libraries
TARGET_LIB_CPU := libnvdsinfer_custom_impl_Yolo.so
TARGET_LIB_CUDA := libnvdsinfer_custom_impl_Yolo_cuda.so

# Object files
CPU_OBJ := $(CPU_SRC:.cpp=.o)
CUDA_OBJ := $(CUDA_SRC:.cu=.o)

# Build all targets
all: $(TARGET_LIB_CPU) $(TARGET_LIB_CUDA)
	@echo "================================================"
	@echo "Build complete!"
	@echo "CPU Parser:  $(TARGET_LIB_CPU)"
	@echo "CUDA Parser: $(TARGET_LIB_CUDA)"
	@echo "Architecture: $(TARGET_DEVICE)"
	@echo "================================================"

# CPU parser library (linked with g++)
$(TARGET_LIB_CPU): $(CPU_OBJ)
	$(CC) -o $@ $^ $(LFLAGS_CPU)
	@echo "✓ CPU parser built successfully"

# CUDA parser library (linked with nvcc)
$(TARGET_LIB_CUDA): $(CUDA_OBJ)
	$(NVCC) -o $@ $^ $(GENCODE_FLAGS) $(LFLAGS_CUDA)
	@echo "✓ CUDA parser built successfully"

# Compile CPU source
%.o: %.cpp
	@echo "Compiling CPU parser: $<"
	$(CC) -c -o $@ $(CFLAGS) $<

# Compile CUDA source
%.o: %.cu
	@echo "Compiling CUDA parser: $<"
	$(NVCC) -c -o $@ $(GENCODE_FLAGS) $(CUFLAGS) $<

# Clean build artifacts
clean:
	rm -f $(CPU_OBJ) $(CUDA_OBJ) $(TARGET_LIB_CPU) $(TARGET_LIB_CUDA)
	@echo "✓ Cleaned build artifacts"

# Install to system (optional)
install: all
	sudo cp $(TARGET_LIB_CPU) /opt/nvidia/deepstream/deepstream/lib/
	sudo cp $(TARGET_LIB_CUDA) /opt/nvidia/deepstream/deepstream/lib/
	@echo "✓ Installed to DeepStream lib directory"

# Print build info
info:
	@echo "Target Device: $(TARGET_DEVICE)"
	@echo "CUDA Path: /usr/local/cuda"
	@echo "DeepStream Path: /opt/nvidia/deepstream/deepstream"
	@echo "Gencode Flags: $(GENCODE_FLAGS)"
	@echo "CPU Linker Flags: $(LFLAGS_CPU)"
	@echo "CUDA Linker Flags: $(LFLAGS_CUDA)"

.PHONY: all clean install info
