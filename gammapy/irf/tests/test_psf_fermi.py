# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.utils.data import get_pkg_data_filename
from ..psf_fermi import FermiPSF

def test_FermiPSF():
    filename = get_pkg_data_filename('../../datasets/fermi/psf.fits')
    psf = FermiPSF.read(filename)


def interactive_test():
    filename = get_pkg_data_filename('../../datasets/fermi/psf.fits')
    psf = FermiPSF.read(filename)
    #psf.plot_containment('fermi_psf_containment.pdf')
    #psf.plot_exposure('fermi_psf_exposure.pdf')
    psf.plot_theta('fermi_psf_theta.pdf')
