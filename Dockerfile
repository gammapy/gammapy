# gammapy docker container
FROM continuumio/miniconda3:latest
RUN apt-get update \
    && apt-get install -y python3-pip \
    && rm -rf /var/lib/apt/lists/*

# define workdir
WORKDIR /usr/src/app/

# copy content
COPY . /usr/src/app/

# create and activate conda env
RUN conda install -c conda-forge mamba \
    && mamba env create -n gammapy -f environment-dev.yml \
    && conda clean --all --yes
RUN echo 'source activate gammapy' >> ~/.bashrc
ENV PATH /opt/conda/envs/gammapy/bin:$PATH

# install gammapy
RUN pip install -e .

# download datasets
RUN gammapy download datasets
ENV GAMMAPY_DATA /usr/src/app/gammapy-datasets

# add external notebook functionality
RUN echo 'alias gammapylab="jupyter lab --no-browser --ip=0.0.0.0 --allow-root"' >> ~/.bashrc
EXPOSE 8888

# define entry
CMD [ "/bin/bash" ]