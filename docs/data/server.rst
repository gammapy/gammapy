.. include:: ../references.txt

.. _obs_server:

Data Server
===========

The ``gammapy-data-manage`` tool expects that data is organised in a certain way,
both locally for end users and on servers for data providers.

This format and some background information is given on this page,
i.e. this is mostly useful info if you'd like to distribute data.

File and folder structure
-------------------------

The server must contain a ``data.yaml`` file, which declares which data
it provides, where it's located and a little bit of info about the data
that can help users decide if they want it.

Here's an example ``data.yaml`` file:

.. code-block:: yaml

    TODO

It is strongly recommended, but not absolutely required, that all data
is stored in a single directory, with the ``data.yaml`` file at the top
level. You can use symlinks if the files are actually distributed in
several directories to make it appear as if they are in one directory.

The main advantage of having everything in one folder is that it's
easy to mirror all data using `rsync`_ to another machine (end-user or server):

.. code-block:: bash

 rsync -uvrl <USERNAME>@<HOSTNAME>:<SERVER_DATA_ROOT_DIR> .


Config files
------------

TODO: probably need a local config file the user can edit, as well as an online
config file with pre-defined remotes

.. code-block:: yaml

    TODO


