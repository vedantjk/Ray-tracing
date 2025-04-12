# Makefile for code_gpu.cu

# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -O3 -arch=sm_70

# Target executable
TARGET = raytrace

# Source files
SRCS = code_gpu_float.cu

# Object files
OBJS = $(SRCS:.cu=.o)

# Phony targets
.PHONY: all clean

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) $(OBJS) -o $@

# Rule to compile .cu files to .o files
%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

# Clean target
clean:
	rm -f $(OBJS) $(TARGET)
