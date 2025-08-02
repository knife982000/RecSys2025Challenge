FROM ubuntu:oracular

#Getting updated packages 
RUN apt-get update 
RUN apt-get upgrade -y
RUN apt-get install -y curl
RUN apt-get install -y git
RUN apt-get install -y zip
RUN apt-get install -y wget
RUN apt-get install -y unzip


#COPY pyproject.toml /root
#Getting Miniconda 
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm -f Miniconda3-latest-Linux-x86_64.sh

#Conda init
ENV PATH="/root/miniconda3/bin":$PATH
RUN conda init bash
RUN echo 'conda activate base' >> ~/.bashrc

#Installing Python packages
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda install -y python=3.11 conda-forge::polars=1.24.0 conda-forge::tqdm conda-forge::matplotlib conda-forge::seaborn conda-forge::scikit-learn conda-forge::scipy conda-forge::jupyterlab conda-forge::ipywidgets
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN conda clean -afy

RUN mkdir /recsys2025
WORKDIR /recsys2025