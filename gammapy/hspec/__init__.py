# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""hspec - Spectral analysis with Sherpa

What is this?
-------------

This is a set of functions and classes for XSPEC-style spectral analysis of
HESS data using Sherpa written by Ignasi and Regis.

TODO: Integrate hspec in Sherpa directly or gammapy.spectrum (probably won't remain a separate sub-package).

Documentation
-------------

Sherpa should be available in the current python installation (e.g. by initializing CIAO, or by installing standalone Sherpa,
http://pysherpa.blogspot.fr/2014/10/how-do-i-use-standalone-build-of-sherpa.html).

Execute the run_fit.py without arguments to get available options.

Minimum usage:
$ ~/sw/Hspec/run_fit.py CrabNebula
(in case of execution without further arguments, user will be interactively prompted for location of files and model to use)

Example of command line usage:
~/sw/Hspec/run_fit.py CrabNebula -i run?????.pha -m powlaw1d+logparabola

In the case above, powlaw1d could represent the source of interest and logparabola a contamination from a nearby source.
Please use --manual to fix parameters.

Use --noplot to just perform the fit (without graphical output).
"""
