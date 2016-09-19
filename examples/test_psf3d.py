"""Short script for testing the PSF3D class"""
from astropy.coordinates import Angle
from astropy.units import Quantity
from gammapy.irf import PSF3D


def load_psf():
    """Load a test PSF."""
    filename = '$GAMMAPY_EXTRA/test_datasets/psf_table_023523.fits.gz'
    print('Reading {}'.format(filename))

    return PSF3D.read(filename)


if __name__ == '__main__':
    """Produce some debugging plots"""
    psf = load_psf()
    print(psf.info())
    psf.plot_psf_vs_rad(theta=Angle(0, 'deg'), energy=Quantity(1, 'TeV'))
    psf.peek()
