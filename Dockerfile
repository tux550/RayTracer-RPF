FROM ubuntu:22.04
MAINTAINER Amit Bakshi <ambakshi@gmail.com>

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -yq && apt-get install -yq \
    build-essential \
    gcc \
    g++ \
    cmake \
    make \
    libpthread-stubs0-dev
ADD . /app
WORKDIR /app/build
RUN cmake -G 'Unix Makefiles' ..
RUN /usr/bin/make -j8 
# > ../build.log 2> ../build.err
# Disable for debuggingq
#RUN ./pbrt_test



