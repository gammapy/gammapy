.. _astro-darkmatter:

****************************************
Dark matter (`gammapy.astro.darkmatter`)
****************************************

.. currentmodule:: gammapy.astro.darkmatter

Introduction
============

The `gammapy.astro.darkmatter` module provides spatial and spectral models for
indirect dark matter searches. This introduction is aimed at people who already
have some experience with dark matter analysis. For a thorough introduction see
e.g. `Cirelli 2014`_.

The spatial distribution of dark matter halos is typically modeled with
radially symmetric profiles. Common profiles are the ones by Navarro, Frenk and
White (NFW) or Einasto for cuspy and an Isothermal or Burkert profile for cored
dark matter distributions (see `gammapy.astro.darkmatter.profiles`).

The spectral models in `gammapy.astro.darkmatter.PrimaryFlux` are based on
`Cirelli et al.  2011`_, who provide tabulated spectra for different
annihilation channels). These models are most commonly used in VHE dark matter
analyses.

Other packages
==============

There are many other packages out there that implement functionality for dark
matter analysis, their capabilities are summarized in the following

GammaLib
--------

The `GammaLib 1.3 release`_ contains radial profile spatial models, including
Dark Matter halo models, such as GModelSpatialRadialProfileDMBurkert,
GModelSpatialRadialProfileDMEinasto, GModelSpatialRadialProfileDMZhao. There is
some discussion about the implementation in `Feature request #1520`_.  From a
superficial search in the GammaLib docs it seems that there are no spectral
models implemented.

FermiST
-------

The Fermi Science tools have a `DMFitFunction`_ with the following XML
serialization format

.. code-block:: xml

    <source name="DM_Example" type="PointSource">
    <spectrum file="$(BASE_DIR)/data/Likelihood/gammamc_dif.dat" type="DMFitFunction">
    <parameter error="1." free="0" max="1.e+5" min="1.e-5" name="norm" scale="1.e+20" value="5.0" />
    <parameter error="1." free="0" max="5000.0" min="0." name="sigmav" scale="1.e-26" value="3.0" />
    <parameter error="1." free="0" max="5000.0" min="1." name="mass" scale="1.0" value="10"/>
    <parameter error="0.1" free="0" max="1.0" min="0.0" name="bratio" scale="1.0" value="1"/>
    <parameter free="0" max="10" min="1" name="channel0" scale="1.0" value="4"/> <parameter free="0" max="10" min="1" name="channel1" scale="1.0" value="1"/>
    </spectrum>
    <spatialModel type="SkyDirFunction">
    <parameter free="0" max="360" min="-360" name="RA" scale="1.0" value="128.8272"/>
    <parameter free="0" max="90" min="-90" name="DEC" scale="1.0" value="-45.1762"/>
    </spatialModel>
    </source>

The `DMFitFunction`_ is only a spectral model and the spatial component is
set using a point source. A spatial template can obviously be used. Utilities
to create such sky maps are for example `fermipy/dmsky`_ but it seems like this
package is basically a collection of spatial models from the literature. There
is also `fermiPy/dmpipe`_ but it also does not seem to implement any spatial
profiles.


The DMFitFunction is also implemented in `fermipy.spectrum.DMFitFunction`_.
It is a spectral model based on `Jeltema & Profuma 2008`_. From a quick look I
didn't see where they get the spectral template from (obviously not `Cirelli et
al. 2011`_) but `DarkSUSY`_ is mentioned in the paper.

DMFitFunction is also implemented in `astromodels`_.

None of the mentioned packages implement the spectral models by `Cirelli et al.  2011`_

CLUMPY
------

`CLUMPY`_ is a package for γ-ray signals from dark matter structures. The core
of the code is the calculation of the line of sight integral of the dark matter
density squared (for annihilations) or density (for decaying dark matter).
CLUMPY is written in C/C++ and relies on the CERN ROOT library. There is no
Python wrapper, as far as I can see. The available dark matter profiles go
beyond what is used in usual VHE analyses. It might be worth looking into this
package for cross checking the functionality in gammapy.

gamLike
-------

`GamLike`_ contains likelihood functions for most leading gamma-ray indirect
searches for dark matter, including Fermi-LAT observations of dwarfs and the
Galactic Centre (GC), HESS observations of the GC, and projected sensitivites
for CTA observations of the GC. It is released in tandem with the `GAMBIT`_
module `DarkBit`_.  DarkBit can be used for directly computing observables and
likelihoods, for any combination of parameter values in some underlying
particle model. GamLike can somehow be use to reproduce HESS results (Section
6.2.2. of the DarkBit paper). But I don't fully understand how.

Using `gammapy.spectrum`
========================

For use cases please go to the tutorial notebooks:

* `astro_dark_matter.html <../../notebooks/astro_dark_matter.html>`__

Reference/API
=============

.. automodapi:: gammapy.astro.darkmatter
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.astro.darkmatter.profiles
    :no-inheritance-diagram:
    :include-all-objects:

.. _GammaLib 1.3 release: http://cta.irap.omp.eu/gammalib-devel/admin/release_history/1.3.html
.. _Feature request #1520:  https://cta-redmine.irap.omp.eu/issues/1520
.. _Cirelli et al. 2011: http://iopscience.iop.org/article/10.1088/1475-7516/2011/03/051/pdf
.. _Cirelli 2014: http://www.marcocirelli.net/otherworks/HDR.pdf
.. _DMFitFunction: https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html#DMFitFunction
.. _fermipy/dmsky: https://github.com/fermiPy/dmsky
.. _fermipy/dmpipe: https://github.com/fermiPy/dmpipe
.. _fermipy.spectrum.DMFitFunction: https://github.com/fermiPy/fermipy/blob/1c2291a4cbdf30f3940a472bcce2a45984c339a6/fermipy/spectrum.py#L504
.. _Jeltema & Profuma 2008: http://iopscience.iop.org/article/10.1088/1475-7516/2008/11/003/meta
.. _astromodels: https://github.com/giacomov/astromodels/blob/master/astromodels/functions/dark_matter/dm_models.py
.. _CLUMPY: http://lpsc.in2p3.fr/clumpy/
.. _DarkSUSY: http://www.darksusy.org/
.. _GamLike: https://bitbucket.org/weniger/gamlike
.. _GAMBIT: https://gambit.hepforge.org/
.. _DarkBit: https://link.springer.com/article/10.1140%2Fepjc%2Fs10052-017-5155-4
