.. include:: ../references.txt

.. _dataformats:

************
Data Formats
************

.. note:: Since November 2015 there is the :ref:`gadf:main-page` project.
    This page contains extra information about which formats we support in
    Gammapy and which class corresponds to which format.

Where available and useful existing standards are used, e.g. for spectral data
the X-ray community has developed the ``PHA``, ``ARF`` and ``RMF`` file formats
and they have developed powerful tools to work with data in that format.

.. _dataformats_overview:

Overview
--------

Here's an overview of the file formats supported by Gammapy and Gammalib:

================= ==================== ================================================= ============================
Type              Format Name          Gammapy                                           Gammalib
================= ==================== ================================================= ============================
Events            EVENTS               `~gammapy.data.EventList`                         GEventList_
----------------- -------------------- ------------------------------------------------- ----------------------------
Effective area    AEFF_2D              `~gammapy.irf.EffectiveAreaTable2D`               `GCTAAeff2D`_
Effective area    ARF                  `~gammapy.irf.EffectiveAreaTable`                 `GCTAAeffArf`_
----------------- -------------------- ------------------------------------------------- ----------------------------
Energy dispersion EDISP_2D             `~gammapy.irf.EnergyDispersion2D`                 `GCTAEdisp2D`_
Energy dispersion RMF                  `~gammapy.irf.EDispKernel`                        `GCTAEdispRMF`_
----------------- -------------------- ------------------------------------------------- ----------------------------
PSF               PSF_2D_GAUSS         `~gammapy.irf.EnergyDependentMultiGaussPSF`       `GCTAPsf2D`_
PSF               PSF_2D_KING          `~gammapy.irf.PSFKing`                            `GCTAPsfKing`_
PSF               no spec available    `~gammapy.irf.TablePSF`                           `GCTAPsfVector`_
PSF               gtpsf_ output        `~gammapy.irf.EnergyDependentTablePSF`            N/A
PSF               psf_table            `~gammapy.irf.PSF3D`                              `GCTAPsfTable`_
PSF               no spec available    N/A                                               `GCTACubePsf`_
----------------- -------------------- ------------------------------------------------- ----------------------------
Background        BACKGROUND_3D        `~gammapy.irf.Background3D`                       `GCTABackground3D`_
Background        BACKGROUND_2D        `~gammapy.irf.Background2D`                       N/A
Background        no spec available    N/A                                               `GCTAModelRadialAcceptance`_
Background        no spec available    N/A                                               `GCTACubeBackground`_
----------------- -------------------- ------------------------------------------------- ----------------------------
Exposure          EXPOSURE_3D          `~gammapy.maps.Map`                               `GCTACubeExposure`_
----------------- -------------------- ------------------------------------------------- ----------------------------
Counts            3D                   `~gammapy.maps.Map`                               `GCTAEventCube`_
Counts            Image                `~gammapy.maps.Map`                               `GSkyMap`_
Counts            PHA                  `~gammapy.spectrum.CountsSpectrum`                `GPha`_
================= ==================== ================================================= ============================

.. _GEventList: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGEventList.html

.. _GCTAEventCube: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGCTAEventCube.html
.. _GSkyMap: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGSkyMap.html
.. _GPha: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGPha.html

.. _GCTABackground3D: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGCTABackground3D.html
.. _GCTAModelRadialAcceptance: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGCTAModelRadialAcceptance.html
.. _GCTACubeBackground: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGCTACubeBackground.html

.. _GCTACubeExposure: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGCTACubeExposure.html

.. _GCTAPsf2D: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGCTAPsf2D.html
.. _GCTAPsfKing: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGCTAPsfKing.html
.. _GCTAPsfVector: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGCTAPsfVector.html
.. _GCTAPsfTable: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGCTAPsfTable.html
.. _GCTACubePsf: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGCTACubePsf.html

.. _GCTAAeff2D: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGCTAAeff2D.html
.. _GCTAAeffArf: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGCTAAeffArf.html

.. _GCTAEdispRMF: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGCTAEdispRmf.html
.. _GCTAEdisp2D: http://cta.irap.omp.eu/gammalib-devel/doxygen/classGCTAEdisp2D.html


Notes
+++++

* The Gammalib docs contain a nice overview of IRFs
  `here <http://cta.irap.omp.eu/gammalib-devel/user_manual/modules/cta_overview.html>`__
  and detailed explanations of some IRF
  `here <http://cta.irap.omp.eu/gammalib-devel/user_manual/modules/cta_irf.html>`__.
* We probably should unify / shorten the IRF class names in Gammapy.
* There's quite a few classes in Gammapy that don't have a well-defined format.
  We should add a way to serialise every class and document the format for easier interop.
* Maybe add info which format is used by Fermi, HESS HD/PA, CTA?
* For every format there should be one or several test data files.
  This could even be in a repo that's shared by Gammalib / Gammapy.
