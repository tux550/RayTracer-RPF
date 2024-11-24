image_name := "pbrt-rpf"
image_version := "0.1"
image_tag := image_name + ":" + image_version

default: build

ensure-build:
    mkdir -p build/

ensure-output:
    mkdir -p output/

clean:
    rm -rf build/

cmake-gen: ensure-build
    cd build/ && cmake -G 'Unix Makefiles' ..

build: ensure-build
    cd build/ && make -j8

run scene="scenes/sample.pbrt": ensure-output
    ./build/pbrt {{scene}}
    mv *.exr output/
    chmod a+rwx -R output

build-image:
    docker build -t {{image_tag}} .

enter-docker:
    docker run -it --rm -w /pbrt -v {{justfile_directory()}}:/pbrt {{image_tag}} /bin/bash

test:
    echo "Nothing yet"
    # ./pbrt_test
