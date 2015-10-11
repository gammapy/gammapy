.. include:: ../../references.txt

.. _tutorials-fermi_psf:

Fermi-LAT PSF Study
===================

This study compares the results of the Fermi Science Tool `gtpsf`_ - which gives
information about the size of the Fermi-LAT PSF with energy - for the Galactic
center `~gammapy.datasets.FermiGalacticCenter` and in the region of the
Vela Pulsar `~gammapy.datasets.FermiVelaRegion`. The plots below compare
the PSF calculated using the Science Tools in these locations, and compare
to the published LAT PSF data for Pass 7 ``P7SOURCEV6`` and reprocessed
Pass 7 ``P7REP_SOURCE_V15`` IRFs at energies between 10 and 300 GeV (the range
of energies for which the calculated and published PSF results are all
available). 68% and 95% PSF containment radii are considered.

.. literalinclude:: fermi_psf_study.py

Note that for the ``P7SOURCEV6`` and ``P7REP_SOURCE_V15`` lines, the data was
extracted by hand and so **a 10% error should be assumed**

.. plot:: tutorials/fermi_psf/plot_68.py
   
.. plot:: tutorials/fermi_psf/plot_95.py
 
The plot for ``P7REP_SOURCE_V15`` is not available online, but ``P7REP_CLEAN_V15``
is very similar to ``P7REP_SOURCE_V15`` which is used for this study.

The plots indicate that ``P7REP_CLEAN_V15`` cuts (which are very similar to
``P7REP_SOURCE_V15`` cuts) were used for the Vela Region data. However, for
the Galactic Center region, ``P7SOURCEV6`` cuts are consistent with the PSF
data, and ``P7REP_CLEAN_V15`` could not have been used here.
 
The published LAT PSF data may be found at:

* `PSF_P7REP_SOURCE_V15 <http://www.slac.stanford.edu/exp/glast/groups/canda/archive/p7rep_v15/lat_Performance_files/cPsfEnergy_P7REP_SOURCE_V15.png>`_
* `PSF_P7SOURCEV6 <http://www.slac.stanford.edu/exp/glast/groups/canda/archive/pass7v6/lat_Performance_files/cPsfEnergy_P7SOURCE_V6.png>`_
