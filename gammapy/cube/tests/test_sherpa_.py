# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing.utils import assert_allclose
from astropy.io import fits
from ...datasets import gammapy_extra
from ...utils.testing import requires_dependency, requires_data
from .. import SkyCube


@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def test_sherpa_crab_fit():
    from sherpa.models import NormGauss2D, PowLaw1D, TableModel, Const2D
    from sherpa.stats import Chi2ConstVar
    from sherpa.optmethods import LevMar
    from sherpa.fit import Fit
    from ..sherpa_ import CombinedModel3D

    filename = gammapy_extra.filename('experiments/sherpa_cube_analysis/counts.fits.gz')
    # Note: The cube is stored in incorrect format
    counts = SkyCube.read(filename, format='fermi-counts')
    cube = counts.to_sherpa_data3d()

    # Set up exposure table model
    filename = gammapy_extra.filename('experiments/sherpa_cube_analysis/exposure.fits.gz')
    exposure_data = fits.getdata(filename)
    exposure = TableModel('exposure')
    exposure.load(None, exposure_data.ravel())

    # Freeze exposure amplitude
    exposure.ampl.freeze()

    # Setup combined spatial and spectral model
    spatial_model = NormGauss2D('spatial-model')
    spectral_model = PowLaw1D('spectral-model')
    source_model = CombinedModel3D(spatial_model=spatial_model, spectral_model=spectral_model)

    # Set starting values
    source_model.gamma = 2.2
    source_model.xpos = 83.6
    source_model.ypos = 22.01
    source_model.fwhm = 0.12
    source_model.ampl = 0.05

    model = 1E-9 * exposure * source_model  # 1E-9 flux factor

    # Fit
    fit = Fit(data=cube, model=model, stat=Chi2ConstVar(), method=LevMar())
    result = fit.fit()

    reference = [0.121556,
                 83.625627,
                 22.015564,
                 0.096903,
                 2.240989]

    assert_allclose(result.parvals, reference, rtol=1E-5)
