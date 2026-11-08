FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch v0.9.0 https://github.com/ggml-org/llama.cpp /root/llama.cpp \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && mkdir -p /root/llama.cpp/build \
    && cd /root/llama.cpp/build \
    && export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    && export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH} \
    && cmake .. \
        -DGGML_CUDA=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
    && make -j$(nproc) \
    && rm /usr/local/cuda/lib64/stubs/libcuda.so.1

RUN git clone --branch v0.9.4 https://github.com/0xShug0/audio.cpp /root/audio.cpp \
    && cd /root/audio.cpp/ \
    && scripts/build_linux.sh --backend cuda --target audiocpp_cli --target audiocpp_server

# llama.cpp server scripts
COPY run-qwen.sh /root/llama.cpp
RUN chmod u+x /root/llama.cpp/run-qwen.sh
RUN mkdir -p /root/audio.cpp/models

# audio.cpp server scripts
COPY server.json /root/audio.cpp/server.json
COPY run-supertonic.sh /root/audio.cpp
RUN chmod u+x /root/audio.cpp/run-supertonic.sh


# Install uv
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# App
COPY . /root/app
WORKDIR /root/app
RUN uv venv && uv sync
RUN chmod u+x run-app.sh
CMD ["./run-app.sh"]
