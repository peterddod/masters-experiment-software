FROM nvcr.io/nvidia/pytorch:21.11-py3

# install various packages, prune or add to as needed
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing && \
    apt-get -y install \
    python3 \
    ninja-build \
    cmake \
    build-essential \
    llvm-dev \
    git \
    tmux \
    python3-pip \
    wget \
    tree \
    ripgrep

# Install Python requirements (uncomment if you have them)
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Setup Juptyer environment
RUN pip3 install jupyter jupyter_contrib_nbextensions yapf
RUN jupyter contrib nbextension install --user; jupyter nbextensions_configurator enable --user
RUN echo 'alias rjup="jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"' > /root/.bashrc
RUN jupyter nbextension enable code_prettify/code_prettify --sys-prefix
RUN jupyter nbextension enable spellchecker/main --sys-prefix
RUN jupyter nbextension enable toggle_all_line_numbers/main --sys-prefix
RUN jupyter nbextension enable execute_time/ExecuteTime --sys-prefix
RUN jupyter nbextension enable notify/notify --sys-prefix
RUN jupyter nbextension enable toc2/main --sys-prefix
RUN jupyter nbextension enable zenmode/main --sys-prefix

WORKDIR /workspace
