FROM debian:bullseye-slim

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        python3-pip \
        python-is-python3 \
        g++ \
        make \
        cmake \
        git \
        texlive-base \
        latexmk \
        texlive-latex-extra \
        tex-gyre \
        librsvg2-bin \
    # && add-apt-repository -y ppa:deadsnakes/ppa \
    # && apt-get update \
    # && apt-get install python3.7 python3.7-dev \
    # && python3 -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1 \
#     && update-alternatives  --set python /usr/bin/python3.6

RUN pip install tox