.. include:: ../references.txt

.. _docker:

Using a docker container
========================

We can work with Gammapy in a virtual docker container. For this you need to have
`Docker <https://www.docker.com/community-edition>`__ installed in your local machine.

The gammapy docker image
------------------------

You can use a docker image that is stored in the `Gammapy DockerHub repository <https://hub.docker.com/u/gammapy/dashboard/>`__.
This docker image is automatically downloaded when executing the `run` commands below. Alternatively, you can build the
docker image locally by your own with the `Dockerfile` that is provided at the root of the code base. In that case just
type `docker build -t gammapy .` at the top level of the code base folder and remove `gammapy/` from the commands below.

Interactive bash session
------------------------

In order to run a Gammapy docker container you may type the code below.

.. code-block:: bash

    $ docker run --name gammapy -it -p 9000:8888 -v ~:/usr/src/app/host gammapy/gammapy

Once the docker container is running you land in the docker shell as a root user. We will use the prompt % to identify the
commands in the docker container shell. Inside the docker container shell you may list the content of the different folders,
create and execute python scripts, or open an IPython session to work with Gammapy.

Jupyter notebooks
-----------------

It is also possible to create and execute notebooks, following the steps below.

1. Enter in the docker container shell and type:

.. code-block:: bash

        % gammapylab

2. Open a web browser and type:

.. code-block:: bash

        http://localhost:9000

3. Copy the token from the docker container shell into the form displayed by the browser.

List content
------------

In a web browser:

.. code-block:: bash

        http://localhost:9000

In the docker container shell:

.. code-block:: bash

        % cd docs/tutorials
        % ls

Copy content
------------

From the docker container to your home directory:

.. code-block:: bash

        % cp -R docs/tutorials host/

From your home directory to the docker container:

.. code-block:: bash

        % cp host/<myfile> .

Using the browser:

.. code-block:: bash

        http://localhost:9000

Sessions
--------

In order to finish your session and stop the docker container:

.. code-block:: bash

        % exit

To recover a previous session re-starting the *gammapy* docker container:

.. code-block:: bash

        $ docker start -ai gammapy
