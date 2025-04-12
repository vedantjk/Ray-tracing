# Makefile for code_omp.c

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -O3 -fopenmp -lm

# Target executable
TARGET = raytrace

# Source files
SRCS = code_omp_float.c

# Object files
OBJS = $(SRCS:.c=.o)

# Phony targets
.PHONY: all clean

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@

# Rule to compile .c files to .o files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean target
clean:
	rm -f $(OBJS) $(TARGET)
