Run Gammapy in Docker container
===============================

What is this?
-------------

These are some notes how to run Gammapy in a Docker containers.

This is useful e.g. to debug travis-ci fails locally, e.g. this one:
https://github.com/gammapy/gammapy/issues/226

If this doesn't work, you can ask travis-ci via email to provide you
with ssh access to a temp VM that you can use to debug the issue.

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

To replay the commands from travis-ci you can make a bash script with
those commands like this:

    wget https://s3.amazonaws.com/archive.travis-ci.org/jobs/46079771/log.txt
    grep '$ ' log.txt | grep -v '# '
    # This output looks OK, but it has a ton of non-visible characters.
    # I did not find a way to get rid of those, but this works:
    # Manually copy the output on the terminal into a gist on github.
    # Then manually copy that gist content into a file log2.txt
    cat log2.txt | cut -c 3- > commands.sh

    # You have to add this at the top
    export TRAVIS_PYTHON_VERSION=3.4
