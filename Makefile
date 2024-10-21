IMAGE_NAME = pbrt-v3
IMAGE_VERSION = 1.0
IMAGE_TAG = $(IMAGE_NAME):$(IMAGE_VERSION)

DEV_CONTAINER_NAME = pbrt-v3-dev

CONTAINER_WORK_DIR = /app
HOST_OUTPUT_DIR = $(shell pwd)/output
CONTAINER_OUTPUT_DIR = $(CONTAINER_WORK_DIR)/output

run:
	./build/pbrt scenes/sample.pbrt
	mv *.exr output/
build-image:
	docker build -t $(IMAGE_TAG) .
shell:
	docker run -it --rm -w $(CONTAINER_WORK_DIR) -v $(HOST_OUTPUT_DIR):$(CONTAINER_OUTPUT_DIR) $(IMAGE_TAG)  /bin/bash