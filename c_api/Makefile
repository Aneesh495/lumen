CC ?= clang
CFLAGS ?= -O3 -Wall -Wextra -pedantic -std=c11 -fPIC
INCLUDES = -I./include

# Detect OS for dynamic library extension
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    DYLIB_EXT = dylib
    DYLIB_FLAGS = -dynamiclib
else
    DYLIB_EXT = so
    DYLIB_FLAGS = -shared
endif

LIB_NAME = liblumen.$(DYLIB_EXT)
BUILD_DIR = build
BIN_DIR = bin

all: $(BUILD_DIR)/$(LIB_NAME) $(BIN_DIR)/test_c_api

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR)/lumen_c.o: src/lumen_c.c include/lumen.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c src/lumen_c.c -o $(BUILD_DIR)/lumen_c.o

$(BUILD_DIR)/$(LIB_NAME): $(BUILD_DIR)/lumen_c.o | $(BUILD_DIR)
	$(CC) $(DYLIB_FLAGS) $(BUILD_DIR)/lumen_c.o -o $(BUILD_DIR)/$(LIB_NAME) -lm

$(BIN_DIR)/test_c_api: test/test_c_api.c $(BUILD_DIR)/lumen_c.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) test/test_c_api.c $(BUILD_DIR)/lumen_c.o -o $(BIN_DIR)/test_c_api -lm

test: $(BIN_DIR)/test_c_api
	./$(BIN_DIR)/test_c_api

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all test clean
