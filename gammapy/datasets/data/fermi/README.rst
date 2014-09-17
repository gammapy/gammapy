Fermi-LAT small data sets
=========================

We add a few small Fermi-LAT data files that we use for unit tests, examples and tutorials.

Parameters
----------

* 5 years of observation time (2008-08-05 to 2013-08-05)
* Event class and IRF: P7SOURCEV6
* Max zenith angle cut: 105 deg
* 10 GeV < Energy < 500 GeV

Files
-----

* ``fermi_counts.fits.gz`` -- Galactic center counts image
* ``psf.fits`` -- Galactic center PSF
* ``gll_iem_v02_cutout.fits`` -- Galactic center diffuse model cube
* ``fermi_exposure.fits.gz`` -- Galactic center exposure cube


Details
-------

Diffuse Model Cube: `gll_iem_v02_cutout.fits`

Fermi LAT data server query parameters:

* Equatorial coordinates (degrees) (266.405,-28.9362)
* Time range (MET)  (239587200,397353600)
* Time range (Gregorian)  (2008-08-05 00:00:00,2013-08-05 00:00:00)
* Energy range (MeV)   (10000,500000)
* Search radius (degrees) 30

Commands:

I produced the `gll_iem_v02_cutout.fits` file using these commands::

   $ wget http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/aux/gll_iem_v02.fit
   $ ftcopy 'gll_iem_v02.fit[330:390,170:190,*]' gll_iem_v02_cutout.fits
   $ fchecksum gll_iem_v02_cutout.fits update+ datasum+

   
Exposure Cube: `fermi_exposure.fits.gz`

Fermi LAT Key Parameters:

* Equatorial coordinates (degrees) (266.405,-28.9362)
* Time range (MET)  (239587200,397353600)
* Time range (Gregorian)  (2008-08-05 00:00:00,2013-08-05 00:00:00)
* Energy range (MeV)   (10000,1000000)
* Search radius (degrees) 30

Commands:

I produced the `fermi_exposure.fits.gz` file using the commands in the executable script ``make.sh`` with the FSSC Fermi Science Tools.
   
