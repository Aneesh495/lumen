# Makefile for Lumen: High-performance 3D Gaussian Splatting Engine

MOJO ?= $(shell which mojo 2>/dev/null || echo ~/.pixi/bin/pixi run mojo)
PYTHON ?= python3
BIN_DIR = bin
SRC_DIR = src
TEST_DIR = tests

.PHONY: all build test clean hygiene bench run-cli run-server

all: build

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

build: $(BIN_DIR)
	@echo "Building Lumen native binary..."
	$(MOJO) build -I . $(SRC_DIR)/main.mojo -o $(BIN_DIR)/lumen
	@echo "Build successful: $(BIN_DIR)/lumen"

test:
	@echo "Executing complete Lumen test suite..."
	bash scripts/verify_all.sh

hygiene:
	@echo "Verifying typography and zero-dash constraint..."
	$(PYTHON) scripts/verify_no_dashes.py

bench:
	@echo "Running performance benchmarks..."
	$(MOJO) -I . benches/bench_rasterizer.mojo
	$(MOJO) -I . benches/bench_radix_sort.mojo
	$(MOJO) -I . benches/bench_spherical_harmonics.mojo
	$(MOJO) -I . benches/bench_bvh.mojo

run-cli: build
	./$(BIN_DIR)/lumen --help

clean:
	rm -rf $(BIN_DIR) build dist *.o *.dylib *.so .mojo_cache
