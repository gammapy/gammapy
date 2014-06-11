# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Fake example spectral data in XSPEC format.

The following files are generated:
- xspec_test_pha.fits
- xspec_test_arf.fits
- xspec_test_rmf.fits
"""
import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
import numpy as np

# EMIN, EMAX = 1e-2, 1e3
# NBIN = 5 * 10
EBOUNDS = np.logspace(-2, 3, 50)


def make_test_arf():
    from gammapy.irf import np_to_arf
    from gammapy.irf import abramowski_effective_area

    effective_area = abramowski_effective_area(np.diff(EBOUNDS))
    arf = np_to_arf(effective_area, EBOUNDS)

    filename = 'xspec_test_arf.fits'
    logging.info('Writing {0}'.format(filename))
    arf.writeto(filename, clobber=True)


def make_test_rmf():
    from gammapy.irf import gauss_energy_dispersion_matrix
    from gammapy.irf import np_to_rmf

    pdf_matrix = gauss_energy_dispersion_matrix(EBOUNDS)
    rmf = np_to_rmf(pdf_matrix, EBOUNDS, EBOUNDS, minprob=1e-6)

    filename = 'xspec_test_rmf.fits'
    logging.info('Writing {0}'.format(filename))
    rmf.writeto(filename, clobber=True)


def plot_rmf():
    import matplotlib.pyplot as plt
    from gammapy.irf import EnergyDispersion

    #filename = 'xspec_test_rmf.fits'
    filename = '/Users/deil/code/gammalib/inst/cta/test/caldb/dc1/rmf.fits'
    #filename = '/Users/deil/work/host/howto/ctools_crab/cta-1dc/data/hess/dummy_s0.1.rmf.fits'
    filename = '/Users/deil/work/host/howto/xspec/Crab/run_rmf61261.fits'
    logging.info('Reading {0}'.format(filename))
    edisp = EnergyDispersion.read(filename)

    print(edisp)
    plt.figure(figsize=(5, 5))
    edisp.plot()

    filename = 'xspec_test_rmf.png'
    logging.info('Writing {0}'.format(filename))
    plt.savefig(filename, dpi=200)


def make_test_pha():
    pass


if __name__ == '__main__':
    #make_test_arf()
    #make_test_rmf()
    #make_test_pha()
    plot_rmf()
