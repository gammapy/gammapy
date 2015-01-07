Run Gammapy in Docker container
===============================

What is this?
-------------

These are some notes how to run Gammapy in a Docker containers.

This is useful e.g. to debug travis-ci fails locally, e.g. this one:
https://github.com/gammapy/gammapy/issues/226

How does this work?
-------------------

To build the docker image::

    time docker build -t cdeil/gammapy-travis .
    docker push cdeil/gammapy-travis

To run the docker image::

    docker pull cdeil/gammapy-travis
    docker run  -t -i cdeil/gammapy-travis:latest bash

Then you can run Numpy or Astropy or Gammapy tests or do whatever you need.
    
References
----------

* https://github.com/ContinuumIO/docker-images/tree/master/miniconda
* https://registry.hub.docker.com/u/continuumio/miniconda/
* https://registry.hub.docker.com/u/cdeil/gammapy-travis/

Notes
-----

One issue i have on my Mac is this
http://stackoverflow.com/questions/26686358/docker-cant-connect-to-boot2docker-because-of-tcp-timeout
and running this command fixes it:
http://stackoverflow.com/a/26804653/498873
