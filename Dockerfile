FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu16.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    graphviz \
    less \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    llvm \
    make \
    openssh-client \
    python-openssl \
    tk-dev \
    tmux \
    unzip \
    vim \
    wget \
    xz-utils \
    zip \
    zlib1g-dev

ENV HOME /root

ENV PYTHON_VERSION 3.7.4
# ENV PYTHON_VERSION 3.6.8
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH

ENV PYENV_ROOT $HOME/.pyenv
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT && \
    $PYENV_ROOT/plugins/python-build/install.sh && \
    /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT && \
    rm -rf $PYENV_ROOT

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
ENV PATH $HOME/.poetry/bin:$PATH

WORKDIR $HOME/penguin-libraries
COPY pyproject.toml poetry.lock poetry.toml ${WORKDIR}/
RUN mkdir ${WORKDIR}/src && touch ${WORKDIR}/src/__init__.py
RUN pip install --upgrade pip setuptools
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN poetry install

COPY . ${WORKDIR}/

