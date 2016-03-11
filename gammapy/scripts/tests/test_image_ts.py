# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import json
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


def create_test_input_file(input_filename, data):
    maps = [data['counts'], data['background'], data['exposure']]
    maps_names = ['On', 'Background', 'ExpGammaMap']

    hdu_list = fits.HDUList()
    for map_field, name in zip(maps, maps_names):
        hdu = fits.ImageHDU(data=map_field, header=data["header"], name=name)
        hdu_list.append(hdu)
    hdu_list.writeto(input_filename)


def init_psf_file(tmpdir):
    psf_filename = str(tmpdir / 'psf.json')

    psf_pars = dict()
    psf_pars['psf1'] = dict(ampl=0.0254647908947033, fwhm=5.8870501125773735)
    psf_pars['psf2'] = dict(ampl=0, fwhm=1E-5)
    psf_pars['psf3'] = dict(ampl=0, fwhm=1E-5)
    json.dump(psf_pars, open(psf_filename, "w"))

    return psf_filename


@requires_dependency('scipy')
@requires_dependency('skimage')
@requires_data('gammapy-extra')
def test_command_line_gammapy_image_ts(tmpdir):
    """Minimal test of gammapy_image_ts using testcase that
    guaranteed to work with compute_ts_map"""
    data = load_poisson_stats_image(extra_info=True)

    kernel = Gaussian2DKernel(2.5, mode='oversample')
    kernel.normalize()
    data['exposure'] = np.ones(data['counts'].shape) * 1E12
    for _, func in zip(['counts', 'background', 'exposure'], [np.nansum, np.nansum, np.mean]):
        data[_] = downsample_2N(data[_], 2, func)

    result = compute_ts_map(data['counts'], data['background'], data['exposure'],
                            kernel)
    for name, order in zip(['ts', 'amplitude', 'niter'], [2, 5, 0]):
        result[name] = np.nan_to_num(result[name])
        result[name] = upsample_2N(result[name], 2, order=order)

    assert_allclose(1705.840212274973, result.ts[99, 99], rtol=1e-3)
    assert_allclose([[99], [99]], np.where(result.ts == result.ts.max()))
    assert_allclose(6, result.niter[99, 99])
    assert_allclose(1.0227934338735763e-09, result.amplitude[99, 99], rtol=1e-2)

    filename = str(tmpdir / 'ts_test_method.fits')
    result.write(filename, header=data['header'])

    input_filename = str(tmpdir / 'input_all.fits.gz')
    output_filename = str(tmpdir / "output.fits")
    output_filename_upsampled2x = str(tmpdir / "output_upsampled2x.fits")
    psf_filename = init_psf_file(tmpdir)

    create_test_input_file(input_filename, data)
    image_ts_main([input_filename,
                   output_filename,
                   "--psf",
                   psf_filename])

    read_console = TSMapResult.read(output_filename)
    for name, order in zip(['ts', 'amplitude', 'niter'], [2, 5, 0]):
            read_console[name] = np.nan_to_num(read_console[name])
            read_console[name] = upsample_2N(read_console[name], 2, order=order)
    read_console.write(output_filename_upsampled2x, header=data['header'])

    method_hdu = fits.open(filename)[0]
    console_hdu = fits.open(output_filename_upsampled2x)[0]

    assert_allclose(method_hdu.data, console_hdu.data, rtol=1e-2, atol=1e-15)

    # check correct work with different scales
    scales_test_list = ['0.025', '0.075', '0.111']
    correct_filename = str(gammapy_extra.dir /
                         'test_datasets/unbundled/poisson_stats_image/correct_output.fits')
    image_ts_main([input_filename,
                   output_filename,
                   "--psf",
                   psf_filename,
                   "--scales",
                   scales_test_list[0],
                   scales_test_list[1],
                   scales_test_list[2]])

    for scale_test in scales_test_list:
        output_filename_ = output_filename.replace('.fits', '_{0}.fits'.format(scale_test))
        correct_filename_ = correct_filename.replace('.fits', '_{0}.fits'.format(scale_test))

        read_console = TSMapResult.read(output_filename_)

        for name, order in zip(['ts', 'amplitude', 'niter'], [2, 5, 0]):
            read_console[name] = np.nan_to_num(read_console[name])
            read_console[name] = upsample_2N(read_console[name], 2, order=order)
        output_filename_upsampled2x_ = output_filename_upsampled2x.replace('.fits',
                                                                           '_{0}.fits'.format(scale_test))
        read_console.write(output_filename_upsampled2x_, header=data['header'])

        correct_hdu = fits.open(correct_filename_)[0]
        console_hdu = fits.open(output_filename_upsampled2x_)[0]

        assert_allclose(correct_hdu.data, console_hdu.data, rtol=1e-2, atol=1e-15)
