CC = gcc

# Compile-time flags
CFLAGS = -Wall -g

# Loadflags
LDFLAGS = -lcunit

SOURCE_DIR=source

# Define the name for the executable
TEST_TARGET=./bin/tests

# Find all .c files in the source directory
ALL_SOURCES = $(shell find "$(SOURCE_DIR)" -type f -name '*.c')

TEST_SOURCES=$(filter-out %/program.c, $(ALL_SOURCES))

# Convert the .c files filenames to .o to give a list of object to build and clean
TEST_OBJECTS=$(TEST_SOURCES:.c=.o)

# The first rule is the one executed when no parameters are fed into the Makefile
test: $(TEST_OBJECTS)
	echo $(TEST_SOURCES)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $(TEST_TARGET)
	
# This is a rule for cleaning up your build by removing the executable
clean:
	rm -f $(TEST_TARGET) $(TEST_OBJECTS)

#compile .c files to .o files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@