FROM continuumio/miniconda3

# LABEL "itsmenatasha"
# ENV GIT_REPO="alpha_mix_active_learning"
WORKDIR /home/alphamix/
# ENV ROOT="${WORKDIR}${GITREPO}/"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && apt-get install -y nano

# RUN git clone https://github.com/pasquale90/${GIT_REPO}.git
# RUN git checkout docker
COPY . ${WORKDIR}

RUN conda env create -n alphamix -f requirements.yml
SHELL ["conda", "run", "-n", "alphamix", "/bin/bash", "-c"]
# SHELL ["conda", "run", "-n", ${CONDA_ENV}, "/bin/bash", "-c"]
# RUN conda init bash
# CMD echo conda activate alphamix > ~/.bashrc

# COPY ./demo.sh .

ENTRYPOINT [ "sh", "-c", "/bin/bash"]
# ENTRYPOINT ["conda", "run", "-n", "alphamix", "python3", "src/server.py"]


# ,"conda","activate",${GIT_REPO}] 
# CMD ["echo","run-$bash demo.sh-to_run_demo_code"]

# RUN mkdir /myvol && echo "hello world" > /myvol/greeting
# VOLUME [ "/myvol" ]

# CMD /bin/bash -c conda activate alphamix
# CMD ["python","welcome.py"]


