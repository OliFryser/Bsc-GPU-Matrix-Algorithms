CC = gcc
NVCC = nvcc

# Compile-time flags
CFLAGS = -g

# Loadflags
LDFLAGS = -lcunit -lcudart

SOURCE_DIR=source

# Define the name for the executable
TEST_TARGET=./bin/test

# Find all .c files in the source directory
ALL_SOURCES_C = $(shell find "$(SOURCE_DIR)" -type f -name '*.c')
ALL_SOURCES_CU = $(shell find "$(SOURCE_DIR)" -type f -name '*.cu')

TEST_SOURCES_C=$(filter-out %/benchmark_runner.c %/benchmark_sanity_check.c, $(ALL_SOURCES_C))
TEST_SOURCES_CU=$(filter-out %/saxpy.cu, $(ALL_SOURCES_CU))

# Convert the .c files filenames to .o to give a list of object to build and clean
TEST_OBJECTS_C=$(TEST_SOURCES_C:.c=.o)
TEST_OBJECTS_CU=$(TEST_SOURCES_CU:.cu=.o)

# The first rule is the one executed when no parameters are fed into the Makefile
test: $(TEST_OBJECTS_C) $(TEST_OBJECTS_CU)
	$(CC) -L/usr/local/cuda/lib64 -o $(TEST_TARGET) $(CFLAGS) $^ $(LDFLAGS)
	
# This is a rule for cleaning up your build by removing the executable
clean:
	rm -f $(TEST_TARGET) $(TEST_OBJECTS_C) $(TEST_OBJECTS_CU)

#compile .c files to .o files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

#compile .c files to .o files
%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@