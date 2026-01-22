# gammapy docker container
FROM continuumio/miniconda3:latest
RUN apt-get update \
    && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# define workdir
WORKDIR /usr/src/app/

# copy content
COPY . /usr/src/app/

# create and activate conda env
RUN conda install -c conda-forge mamba \
    && mamba env create -n gammapy -f environment-dev.yml \
    && conda clean --all --yes

# add jupyter lab alias and env activation to .bashrc config file
RUN echo 'alias gammapylab="jupyter lab --no-browser --ip=0.0.0.0 --allow-root"\nconda activate gammapy\n' >> ~/.bashrc
EXPOSE 8888

# Make RUN commands use the gammapy environment
SHELL ["conda", "run", "-n", "gammapy", "/bin/bash", "-c"]

# install gammapy
RUN pip install -e .

# download datasets
RUN gammapy download datasets
ENV GAMMAPY_DATA /usr/src/app/gammapy-datasets

# define entry
CMD [ "/bin/bash" ]