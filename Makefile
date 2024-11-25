# Variables
image_name := pbrt-rpf
image_version := 0.1
image_tag := $(image_name):$(image_version)

# Default target
default: build

# Directories
ensure-build:
	mkdir -p ./build/

ensure-output:
	mkdir -p output/

# Cleaning
clean:
	rm -rf build/

# Generate build files with CMake
cmake-gen: ensure-build
	cd build && cmake -G 'Unix Makefiles' ..

# Build
build: ensure-build
	cd build && make -j8

# Run
run: ensure-output
	./build/pbrt scenes/sample.pbrt
	mv *.exr output/
	chmod a+rwx -R output

# Docker build
build-image:
	docker build -t $(image_tag) .

# Enter Docker container
shell:
	docker run -it --rm -w /pbrt -v $(shell pwd):/pbrt $(image_tag) /bin/bash

# Tests (placeholder)
test:
	@echo "Nothing yet"
	# ./pbrt_test
