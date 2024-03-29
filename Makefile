CC = gcc
NVCC = nvcc

CUFLAGS = -g

# Compile-time flags
CFLAGS = -Wall -g

# Loadflags
LDFLAGS = -lcunit -lcudart -lm

SOURCE_DIR=source

# Define the name for the executable
TEST_TARGET=./bin/test

# Find all .c files in the source directory
ALL_SOURCES_C = $(shell find "$(SOURCE_DIR)" -type f -name '*.c' -not -path "*/2DMatrixAlgorithms/*" -not -path "*/DiagnosticTools/*")
ALL_SOURCES_CU = $(shell find "$(SOURCE_DIR)" -type f -name '*.cu' -not -path "*/2DMatrixAlgorithms/*")

TEST_SOURCES_C=$(filter-out %/benchmark_runner.c %/benchmark_sanity_check.c %/cpu_diagnostic.c, $(ALL_SOURCES_C))
TEST_SOURCES_CU=$(filter-out %/saxpy.cu, $(ALL_SOURCES_CU))

# Convert the .c files filenames to .o to give a list of object to build and clean
TEST_OBJECTS_C=$(TEST_SOURCES_C:.c=.o)
TEST_OBJECTS_CU=$(TEST_SOURCES_CU:.cu=.o)

DIAGNOSTIC_SOURCES = $(SOURCE_DIR)/DiagnosticTools/cpu_diagnostic.c
DIAGNOSTIC_OJBECTS= $(DIAGNOSTIC_SOURCES:.c=.o)

bench:
	python3 $(SOURCE_DIR)/AutomationAndBenchmarking/benchmark.py

# The first rule is the one executed when no parameters are fed into the Makefile
test: $(TEST_OBJECTS_C) $(TEST_OBJECTS_CU)
	$(CC) -L/usr/local/cuda/lib64 -o $(TEST_TARGET) $(CFLAGS) $^ $(LDFLAGS)
	
# This is a rule for cleaning up your build by removing the executable
clean:
	rm -f $(TEST_TARGET) $(TEST_OBJECTS_C) $(TEST_OBJECTS_CU) $(DIAGNOSTIC_OJBECTS) $(SOURCE_DIR)/DiagnosticTools/cpu_operations.o

cpu_diagnostic: $(DIAGNOSTIC_OJBECTS)
	$(CC) -O0 -c $(SOURCE_DIR)/DiagnosticTools/cpu_operations.c -o $(SOURCE_DIR)/DiagnosticTools/cpu_operations.o
	$(CC) -o bin/cpu_diagnostic $(CFLAGS) $^ $(SOURCE_DIR)/DiagnosticTools/cpu_operations.o

#compile .c files to .o files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

#compile .c files to .o files
%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@