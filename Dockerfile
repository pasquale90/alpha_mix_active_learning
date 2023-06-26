FROM python:3.10

WORKDIR /home/alphamix/

# RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && apt-get install -y nano
RUN apt-get update -y \ 
    && apt-get install -y wget nano \ 
    && apt-get clean

## CONDA INSTALLATION --> use the latest Anaconda version for linux from their official website. Google it buddy.
RUN rm -rf /opt/conda && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy


## ADD CONDA PATH TO LINUX PATH 
ENV PATH /opt/conda/bin:$PATH


# RUN git clone https://github.com/pasquale90/${GIT_REPO}.git
# RUN git checkout docker
COPY . ${WORKDIR}


## CREATE CONDA ENVIRONMENT USING YML FILE
RUN conda update conda 
RUN conda env create -n alphamix -f requirements.yml
# RUN conda env create -f requirements.yml

    
## ADD CONDA ENV PATH TO LINUX PATH 
ENV PATH /opt/conda/envs/alphamix/bin:$PATH
ENV CONDA_DEFAULT_ENV alphamix
# make sure to put your env name in place of myconda

## MAKE ALL BELOW RUN COMMANDS USE THE NEW CONDA ENVIRONMENT
SHELL ["conda", "run", "-n", "alphamix", "/bin/bash", "-c"]


ENTRYPOINT [ "python", "main.py"]
# https://stackoverflow.com/a/67059519/15842840