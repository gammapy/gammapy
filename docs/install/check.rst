.. _install-check:

Check Gammapy installation
==========================

How to run checks
-----------------

To check if Gammapy is correctly installed, start up python or ipython,
import Gammapy and run the unit tests::

   $ python -c 'import gammapy; gammapy.test()'

To check if the Gammapy command line tools are on your ``$PATH`` try this::

   $ gammapy-info --tools

To check which dependencies of Gammapy you have installed::

   $ gammapy-info --dependencies

.. _install-issues:

Common issues
-------------

If you have an issue with Gammapy installation or usage, please check
this list. If your issue is not adressed, please send an email to the
mailing list.

- Q: I get an error mentioning something (e.g. Astropy) isn't available,
  but I did install it.

  A: Check that you're using the right ``python`` and that your
  ``PYTHONPATH`` isn't pointing to places that aren't appropriate
  for this Python (usually it's best to not set it at all)
  using these commands::

      which python
      echo $PYTHONPATH
      python -c 'import astropy'

