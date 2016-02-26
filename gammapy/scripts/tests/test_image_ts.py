# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing.utils import assert_allclose, assert_equal
from astropy.convolution import Gaussian2DKernel
from ...utils.testing import requires_dependency, requires_data
from ...detect import compute_ts_map, TSMapResult
from ...datasets import load_poisson_stats_image
from ...image.utils import upsample_2N, downsample_2N
from ..image_ts import image_ts_main
from astropy.io import fits
from ...datasets.core import gammapy_extra


@requires_dependency('scipy')
@requires_dependency('skimage')
@requires_data('gammapy-extra')
def test_command_line_gammapy_image_ts(tmpdir):
    """Minimal test of gammapy_image_ts using testcase that
    guaranteed to work with compute_ts_map"""
    data = load_poisson_stats_image(extra_info=True)
    kernel = Gaussian2DKernel(2.5)
    data['exposure'] = np.ones(data['counts'].shape) * 1E12
    for _, func in zip(['counts', 'background', 'exposure'], [np.nansum, np.nansum, np.mean]):
        data[_] = downsample_2N(data[_], 2, func)

    result = compute_ts_map(data['counts'], data['background'], data['exposure'],
                            kernel)
    for name, order in zip(['ts', 'amplitude', 'niter'], [2, 5, 0]):
        result[name] = np.nan_to_num(result[name])
        result[name] = upsample_2N(result[name], 2, order=order)

    filename = str(tmpdir / 'ts_test_method.fits')
    result.write(filename, header=data['header'])

    maps = [data['counts'], data['background'], data['exposure']]
    maps_names = ['On', 'Background', 'ExpGammaMap']

    input_filename = str(tmpdir / 'input_all.fits.gz')
    output_filename = str(tmpdir / "output.fits")
    output_filename_upsampled2x = str(tmpdir / "output_upsampled2x.fits")
    hdu_list = fits.HDUList()
    for map_field, name in zip(maps, maps_names):
        hdu = fits.ImageHDU(data=map_field, header=data["header"], name=name)
        hdu_list.append(hdu)
    hdu_list.writeto(input_filename)

    image_ts_main([input_filename,
                   output_filename,
                   "--psf",
                   str(gammapy_extra.dir /
                       'test_datasets/unbundled/poisson_stats_image/psf.json')])

    # need to upsample, same as with compute_ts_map
    read_console = TSMapResult.read(output_filename)
    for name, order in zip(['ts', 'amplitude', 'niter'], [2, 5, 0]):
        read_console[name] = np.nan_to_num(read_console[name])
        read_console[name] = upsample_2N(read_console[name], 2, order=order)
    result.write(output_filename_upsampled2x, header=data['header'])

    console_hdu = fits.open(output_filename_upsampled2x)[0]
    method_hdu = fits.open(filename)[0]

    assert_equal(method_hdu.data, console_hdu.data)
