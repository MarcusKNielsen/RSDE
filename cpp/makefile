# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Iinclude -std=c++17 -Wall -Wextra -O2

# Source and object directories
SRC_DIR = src
OBJ_DIR = obj

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# Executable name
TARGET = main_program

# Default target
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(OBJS) main.o
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule to compile main.o
main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o $@

# Rule to compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Ensure the object directory exists
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean build files
clean:
	rm -rf $(OBJ_DIR) *.o $(TARGET)

.PHONY: all clean
