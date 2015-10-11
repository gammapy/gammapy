.. _obs:

*************************************
Observation handling  (`gammapy.obs`)
*************************************

.. currentmodule:: gammapy.obs

Introduction
============

`gammapy.obs` contains methods to handle observations.

In TeV astronomy an observation (a.k.a. a run) means pointing the telescopes at some
position on the sky (fixed in celestial coordinates, not in horizon coordinates)
for a given amount of time (e.g. half an hour) and switching the central trigger on.

The total dataset for a given target will usually consist of a few to a few 100 runs
and some book-keeping is required when running the analysis.

Getting Started
===============

Gammapy contains command line tools to manage data and work with subsets of observations.
This allows you to be up and running quickly and to focus on analysis.

* ``gammapy-data-manage`` -- Manage data locally and on servers
* ``gammapy-data-browse`` -- A web app to browse local data (stats and quick look plots)
* ``gammapy-obs-select`` -- Select observations of interest for a given analysis (TODO: rename, currently called ``gammapy-find-obs``)
* ``gammapy-obs-group`` -- Group observations (TODO: implement)

Download data
-------------

List which data you have available locally:

.. code-block:: bash

   $ gammapy-data-manage status

If you're a H.E.S.S. member you can download data like this:

.. code-block:: bash

   $ gammapy-data-manage status

For information on how to distribute data via a data server, see :ref:`obs_server`.

Simulate data
-------------

If not, you can simulate some data to have something to play around with:

.. code-block:: bash

   $ gammapy-data-manage simulate hess01

Other relevant pages: :ref:`datasets_obssim`

Browse data
-----------

We have a web app that lets you browse the local data via a graphical user interface (GUI) in your web browser:

.. code-block:: bash

   $ gammapy-data-browse

This is mostly useful for data producers and experts, not so much for end users.
We plan to add similar web apps for analysts to make it easy to browse analysis
inputs and results for a given target.

Select observations
-------------------

Once you have

.. code-block:: bash

   $ gammapy-data-manage status


Observatory locations
---------------------

Gammapy contains the locations of gamma-ray telescopes:

.. code-block:: python

   >>> from gammapy.obs import observatory_locations
   >>> observatory_locations.HESS
   <EarthLocation (7237.152530011689, 2143.7727767623487, -3229.3927009565496) km>
   >>> print(observatory_locations.HESS.geodetic)
   (<Longitude 16.500222222222224 deg>, <Latitude -23.271777777772456 deg>, <Quantity 1835.0 km>)

This can be convenient e.g. for observation planning, or to transform between Alt-Az and RA-DEC coordinates.

TODO: We should probably update this to use the `astroplan.Observer` class,
which contains a similar observatory lookup database via `astroplan.Observer.at_site`.


Using `gammapy.obs`
===================

If you'd like to learn more about using `gammapy.obs`, read the following sub-pages:

.. toctree::
   :maxdepth: 1

   find_observations
   observation_grouping
   server

Reference/API
=============

.. automodapi:: gammapy.obs
    :no-inheritance-diagram:
