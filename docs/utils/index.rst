.. _utils:

*************************************************
Utility functions and classes (``gammapy.utils``)
*************************************************

Introduction
============

``gammapy.utils`` is a collection of utility functions that are used in many places
or don't fit in one of the other packages.

Since the various sub-modules of ``gammapy.utils`` are mostly unrelated,
they are not imported into the top-level namespace.
Here are some examples of how to import functionality from the ``gammapy.utils``
sub-modules:

.. code-block:: python

   from gammapy.utils.random import sample_sphere
   sample_sphere(size=10)

   from gammapy.utils import random
   random.sample_sphere(size=10)   


Reference/API
=============

.. automodapi:: gammapy.utils.coordinates
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.const
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.fits
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.region
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.root
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.random
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.distributions
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.pyfact
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.scripts
    :no-inheritance-diagram:

.. automodapi:: gammapy.utils.testing
    :no-inheritance-diagram:
    