.. _dataformats_maps:

Maps
====

* ``counts``
* ``background`` --- measured or modeled background estimate (counts per bin)
* ``exposure`` --- gamma-ray exposure
  Exposure maps should contain the following header keywords:
  * ``EXP_GAMMA`` --- power law spectral index assumed in the exposure computation
  * ``UNITS`` --- require cm^-2 sec^-1 or be flexible
  * ``E_MIN``

TODO:
* exposure maps should have:

