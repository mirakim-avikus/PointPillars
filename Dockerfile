FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Some networks TLS-intercept traffic from Docker's bridge subnet, which
# breaks apt's handshake with this HTTPS-only repo. We don't need it anyway:
# everything below comes from the plain-HTTP Ubuntu archive/security mirrors,
# and CUDA itself is already baked into the base image.
RUN rm -f /etc/apt/sources.list.d/cuda*.list

RUN apt-get update && apt-get install -y \
    python3.8 python3-pip python3.8-dev python3.8-distutils \
    git wget curl build-essential cmake libgl1-mesa-glx ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Some networks TLS-inspect all outbound HTTPS (e.g. a corporate SASE gateway),
# presenting their own CA instead of the real one. If yours does, see
# certs/README.md - drop your network's CA at certs/HD_Groups_CA.crt (gitignored,
# never committed) before building. certs/00-placeholder.crt is an always-present
# empty file so this glob has something to match even when no real cert is
# needed; update-ca-certificates just skips it with a harmless warning.
# pip doesn't use the OS trust store by default, so it needs pointing at it
# explicitly too.
COPY certs/*.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

RUN apt-get update && apt-get install -y libglib2.0-0

RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip setuptools==58.0.4

RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install \
    numba==0.48.0 \
    numpy==1.19.5 \
    open3d==0.14.1 \
    opencv_python==4.5.5.62 \
    PyYAML==6.0 \
    tensorboard \
    tqdm==4.62.3

WORKDIR /workspace

