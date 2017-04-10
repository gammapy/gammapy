"""
Example to run the CWT wavelet transform method.
"""
import numpy as np
from numpy.testing import assert_allclose
from gammapy.detect import CWT, CWTData, CWTKernels
from gammapy.image import SkyImage


def make_fermi_data():
    filename = '$GAMMAPY_EXTRA/datasets/fermi_survey/all.fits.gz'

    image = SkyImage.read(filename, hdu='COUNTS')
    background = SkyImage.read(filename, hdu='BACKGROUND')

    return dict(image=image, background=background)


def make_poisson_data():
    from gammapy.datasets import load_poisson_stats_image
    filename = load_poisson_stats_image(return_filenames=True)
    image = SkyImage.read(filename)
    background = SkyImage.read(filename)
    background.data = np.ones_like(image.data, dtype=float)

    return dict(image=image, background=background)


def run_cwt():
    data = make_poisson_data()

    cwt_kernels = CWTKernels(n_scale=2,
                             min_scale=3.0,
                             step_scale=2.6,
                             old=False)
    cwt = CWT(kernels=cwt_kernels,
              significance_threshold=2.,
              keep_history=True)
    cwt_data = CWTData(counts=data['image'],
                       background=data['background'],
                       n_scale=cwt_kernels.n_scale)

    cwt.analyze(data=cwt_data)
    return cwt_data


if __name__ == '__main__':
    # import logging
    # logging.basicConfig(level=logging.INFO)
    result = run_cwt()
    # assert_allclose(result._approx.sum(), 0.)
