"""
Example to run the CWT wavelet transform method.
"""
import numpy as np
from gammapy.detect import CWT


def make_test_data():
    shape = (200, 100)
    background = np.ones(shape, dtype=float)
    image = background.copy()
    image[100, 50] += 10

    return dict(image=image, background=background)


def make_fermi_data():
    import os
    from astropy.io import fits
    filename = os.environ['GAMMAPY_EXTRA'] + '/datasets/fermi_survey/all.fits.gz'
    image = fits.getdata(filename, hdu='COUNTS').astype(float)
    background = fits.getdata(filename, hdu='BACKGROUND').astype(float)
    header = fits.getheader(filename, hdu='COUNTS')

    return dict(image=image, background=background, header=header)


def run_cwt(data):
    cwt = CWT(nscales=2, min_scale=6.0, scale_step=1.3)
    data['background'] = 1. * np.ones_like(data['background'], dtype=float)
    cwt.set_data(data['image'], data['background'], data['header'])
    cwt.iterative_filter_peak(nsigma=100, niter=5)
    return cwt


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    data = make_fermi_data()
    result = run_cwt(data)
    result.save_results('cwt-test.fits', overwrite=True)
    import IPython; IPython.embed()
