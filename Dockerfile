FROM ubuntu:24.04
MAINTAINER Amit Bakshi <ambakshi@gmail.com>

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -yq && apt-get install -yq \
    build-essential \
    gcc \
    g++ \
    cmake \
    make \
    libpthread-stubs0-dev \
    just
ADD . /app
WORKDIR /app/build
# RUN ./pbrt_test
