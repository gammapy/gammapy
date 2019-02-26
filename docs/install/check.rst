.. _install-check:

Check Gammapy installation
==========================

How to run checks
-----------------

To check if Gammapy is correctly installed, start up python or ipython, import
Gammapy and run the unit tests::

    $ python -c 'import gammapy; gammapy.test()'

To check which dependencies of Gammapy you have installed::

    $ gammapy info --dependencies

.. _install-issues:

Common issues
-------------

If you have an issue with Gammapy installation or usage, please check this list.
If your issue is not addressed, please send an email to the mailing list.

- Q: I get an error mentioning something (e.g. Astropy) isn't available,
  but I did install it.

  A: Check that you're using the right ``python`` and that your
  ``PYTHONPATH`` isn't pointing to places that aren't appropriate
  for this Python (usually it's best to not set it at all)
  using these commands::

      which python
      echo $PYTHONPATH
      python -c 'import astropy'


Known issues
------------

- **Astropy <3.0.5 and Matplotlib 3.0**

There is an incompatibility between WCSAxes and Matplotlib 3.0, which causes
the inline plotting in notebooks to fail. The issue as well as a temporary workaround is
described `here <https://github.com/gammapy/gammapy/issues/1843#issuecomment-435909533>`__.
