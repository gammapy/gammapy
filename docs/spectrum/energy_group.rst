.. include:: ../references.txt

.. _spectrum_energy_group:

************************
Spectrum energy grouping
************************

Introduction
============

These are some notes on spectrum energy grouping in Gammapy.

The main application is to compute energy binnings and then compute flux points.

This is work in progress, feedback welcome!


Other packages
==============

HAP FitSpectrum
---------------

HAP has a ``Spectrum`` class, which is a vector of ``SpectrumStats``, which contains
a few numbers (``nOn``, ``nOff``, ``exposureOn``, ``recoAreaTimeOn``, ``liveTime``)
that can be summed when grouping energy bins or observations
and methods to compute derived quantities (``excess``, significance``).

To compute a grouping, the ``Spectrum::Rebin`` method is used, which
returns a re-binned ``Spectrum`` object and has these options:

* ``algorithm`` ``--RebinAlgo`` (string) -- {'NONE', 'Regroup', 'MinSignif', 'MinOnEvents'}
* ``parameter`` ``--RebinParameters`` (float) -- one single parameter, used differently by each algorithm

The adaptive rebinning methods are implemented via ``Spectrum::PairBinsInRange`` left to right.

To select the total energy range, the following options are available:

* Default is the range of bins that covers all counts (``n_on > 0`` or also ``n_off > 0``?)?
  Note that safe energy range has been applied by zeroing out counts and exposure in bins below and above the safe range.
* ``--Emin`` and ``--Emax``
* ``--Min-Livetime-Fraction`` relative fraction of livetime (compared to max) to get rid of low-exposure bins
  in stacked spectra

* https://bitbucket.org/hess_software/flux/src/master/include/Spectrum.hh
* https://bitbucket.org/hess_software/flux/src/master/include/SpectrumStats.hh
* https://bitbucket.org/hess_software/flux/src/master/src/Spectrum.C

Sherpa
------

Sherpa has some spectral grouping functionality

* http://cxc.harvard.edu/sherpa/ahelp/group_sherpa.html
* http://cxc.harvard.edu/sherpa/threads/pha_regroup/
* http://cxc.harvard.edu/sherpa/threads/setplot_manual/
* http://cxc.harvard.edu/ciao/ahelp/dmgroup.html

In the Sherpa Python package this seems to be exposed mainly via the
``sherpa.astropy.data.DataPHA`` class and it's ``group_*`` methods,
that all call into the C ``grplib`` library eventually via a Python
C extension.

* https://github.com/sherpa/sherpa/blob/master/sherpa/astro/data.py
* https://github.com/sherpa/sherpa/tree/master/extern/grplib-4.9

Overall I find the documentation and code not very accessible,
and instead of trying to figure out if we can coerce it to do
all the spectral grouping algorithms we want, for now
I'll go ahead and re-implement grouping via simple Python functions
and classes in Gammapy.

Gammapy Design
==============

Existing functionality
----------------------

The `~gammapy.spectrum.calculate_flux_point_binning` function:

* takes a `~gammapy.spectrum.SpectrumObservationList` and ``min_signif`` as input
* stacks it into a `~gammapy.spectrum.SpectrumObservation` object
* takes the safe energy threshold min and max as range.
* Goes left to right to group adaptively for minimum significance,
  calling `gammapy.data.ObservationStats.stack` to compute stats for grouped bins.
* Returns the grouping as an energy_bounds quantity array.


The `gammapy.spectrum.DifferentialFluxPoints.compute` method computes the grouping
and then computes flux points with that grouping.
The implementation is complex, because it fiddles with ``eps`` to
re-compute the group ID vector from EBOUNDS.

* `gammapy.spectrum.SpectrumObservation`
    * ``total_stats`` -- an `~gammapy.data.ObservationStats`
    * ``stats_table`` -- a table with `~gammapy.data.ObservationStats` for each bin

New proposal
------------

I'd like to implement a little toolbox replicating what HAP FitSpectrum does and more.
Not all the bugs though, it shall be correct and well-tested.

* The output should be a `GROUP_ID` vector or a ``SpectrumEnergyGroupMaker`` object
  (that would be a nice place to attach debug info, print output and plots)
* For the input I'm not sure.
    * Maybe a ``Table`` from  `gammapy.spectrum.SpectrumObservation.stats_table` to have loose coupling?
    * Or a stacked `gammapy.spectrum.SpectrumObservation` object?
    * Or a `~gammapy.spectrum.SpectrumObservationList` object?
* I'm not sure how to structure the code yet and what API to use.
  Ideally it should be simple to use, yet extensible with user-defined methods.
  Maybe it's OK to just have a few pre-baked methods and users that want something different
  will have to write their own function or wrap or sub-class the pre-baked class?

Gammapy Examples
================

Some brainstorming how I'd like to compute spectrum energy groupings as a user.

The main point is to figure out which classes we want and how to configure and run the computation
and return the results.

Let's say we have a `SpectrumObservation` and / or stats summary table::

    obs = SpectrumObservation.read('$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits')
    table = obs.stats_table()

The first step is always to create a ``SpectrumEnergyGroupMaker`` object like this::

    seg = SpectrumEnergyGroupMaker(obs=obs)

TODO: Should we take an ``obs`` object here or a ``table``?
A table would be more loosely coupled, but an ``obs`` might have convenient functionality?

Then one runs some of the ``compute_*`` methods on it, usually::

    seg.compute_range_<method>()
    seg.compute_groups_<method>()

Accessing results always goes like this::

    print(seg) # print summary info
    seg.plot() # make debug plots

    >>> seg.groups.to_group_table()
    # Table with one group per row

    >>> table = seg.groups.to_total_table()
    # Table with one energy bin per row and an energy group index column
    >>> table['energy_group_idx'] # This is the vector defining the grouping

    # Not implemented yet:
    # seg.energy_group_idx # the group ID Numpy array, the main result
    seg.energy_bounds # the energy bounds array (EnergyBounds object)

See the `gammapy.spectrum.SpectrumGroups.to_group_table`
and `gammapy.spectrum.SpectrumGroups.to_total_table` docstrings for more info
on what results are available.

The flux point computation should take either the ``seg.groups.to_total_table()`` table
or the ``seg.groups.to_group_table()`` table as input. Both contain the grouping info
as integer bin indices, so that the flux point computation doesn't have to fiddle with
float energy bounds.

User-supplied energy binning
----------------------------

For a given user-supplied energy binning::

    seg.compute_range_safe() # apply safe energy range
    ebounds = [0.3, 1, 3, 10, 30] * u.TeV
    seg.compute_groups_fixed(ebounds=ebounds)

Adaptive binning
----------------

Here's an example how to run the default HAP FitSpectrum method (min sigma, left to right)::

    seg.compute_range_safe() # uses obs to set the safe energy range
    seg.compute_groups_adaptive(quantity='sigma', threshold=2.0)


Other examples
--------------

...

Other API
---------

Should we expose some pre-baked common energy grouping options
in the API that does the flux point computation, for convenience?

Most users will not have to / want to fiddle with this much.

Gammapy implementation
======================

Astropy table and pandas dataframe has some groupby functionality
that could be useful to compute aggregate stats (e.g. sum and mean)
for groups of bins or anything via ``apply``:

* http://docs.astropy.org/en/stable/table/operations.html#binning
* http://pandas.pydata.org/pandas-docs/stable/groupby.html
