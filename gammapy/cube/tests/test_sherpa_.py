# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing.utils import assert_allclose
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from ...datasets import gammapy_extra
from ...irf import EnergyDispersion
from ...utils.testing import requires_dependency, requires_data
from .. import SkyCube
from ..sherpa_ import NormGauss2DInt


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


@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def testCombinedModel3DInt():
    from sherpa.models import PowLaw1D, TableModel
    from sherpa.estmethods import Covariance
    from sherpa.optmethods import NelderMead
    from sherpa.stats import Cash
    from sherpa.fit import Fit
    from ..sherpa_ import CombinedModel3DInt

    # Set the counts
    filename = gammapy_extra.filename('test_datasets/cube/counts_cube.fits')
    counts_3d = SkyCube.read(filename)
    cube = counts_3d.to_sherpa_data3d(dstype='Data3DInt')

    # Set the bkg
    filename = gammapy_extra.filename('test_datasets/cube/bkg_cube.fits')
    bkg_3d = SkyCube.read(filename)
    bkg = TableModel('bkg')
    bkg.load(None, bkg_3d.data.value.ravel())
    bkg.ampl = 1
    bkg.ampl.freeze()

    # Set the exposure
    filename = gammapy_extra.filename('test_datasets/cube/exposure_cube.fits')
    exposure_3d = SkyCube.read(filename)
    i_nan = np.where(np.isnan(exposure_3d.data))
    exposure_3d.data[i_nan] = 0
    # In order to have the exposure in cm2 s
    exposure_3d.data = exposure_3d.data * 1e4

    # Set the mean psf model
    filename = gammapy_extra.filename('test_datasets/cube/psf_cube.fits')
    psf_3d = SkyCube.read(filename)

    # Setup combined spatial and spectral model
    spatial_model = NormGauss2DInt('spatial-model')
    spectral_model = PowLaw1D('spectral-model')
    source_model = CombinedModel3DInt(use_psf=True, exposure=exposure_3d, psf=psf_3d,
                                      spatial_model=spatial_model, spectral_model=spectral_model)

    # Set starting values
    center = SkyCoord.from_name("Crab").galactic
    source_model.gamma = 2.2
    source_model.xpos = center.l.value
    source_model.ypos = center.b.value
    source_model.fwhm = 0.12
    source_model.ampl = 1.0

    # Fit
    model = bkg + 1E-11 * (source_model)
    fit = Fit(data=cube, model=model, stat=Cash(), method=NelderMead(), estmethod=Covariance())
    result = fit.fit()

    # TODO: The fact that it doesn't converge to the right Crab postion is due to the dummy psf
    reference = [184.2009249783969,
                 -6.1708090667404374,
                 5.1433080482386968,
                 0.078883847523875297,
                 2.2778084618363268]

    assert_allclose(result.parvals, reference, rtol=1E-5)


@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def testCombinedModel3DIntConvolveEdisp():
    from sherpa.models import PowLaw1D, TableModel
    from sherpa.estmethods import Covariance
    from sherpa.optmethods import NelderMead
    from sherpa.stats import Cash
    from sherpa.fit import Fit
    from ..sherpa_ import CombinedModel3DIntConvolveEdisp

    # Set the counts
    filename = gammapy_extra.filename('test_datasets/cube/counts_cube.fits')
    counts_3d = SkyCube.read(filename)
    cube = counts_3d.to_sherpa_data3d(dstype='Data3DInt')

    # Set the bkg
    filename = gammapy_extra.filename('test_datasets/cube/bkg_cube.fits')
    bkg_3d = SkyCube.read(filename)
    bkg = TableModel('bkg')
    bkg.load(None, bkg_3d.data.value.ravel())
    bkg.ampl = 1
    bkg.ampl.freeze()

    # Set the exposure
    filename = gammapy_extra.filename('test_datasets/cube/exposure_cube_etrue.fits')
    exposure_3d = SkyCube.read(filename)
    i_nan = np.where(np.isnan(exposure_3d.data))
    exposure_3d.data[i_nan] = 0
    # In order to have the exposure in cm2 s
    exposure_3d.data = exposure_3d.data * 1e4

    # Set the mean psf model
    filename = gammapy_extra.filename('test_datasets/cube/psf_cube_etrue.fits')
    psf_3d = SkyCube.read(filename)

    # Set the mean rmf
    filename = gammapy_extra.filename('test_datasets/cube/rmf.fits')
    rmf = EnergyDispersion.read(filename)

    # Setup combined spatial and spectral model
    spatial_model = NormGauss2DInt('spatial-model')
    spectral_model = PowLaw1D('spectral-model')
    dimensions = [exposure_3d.data.shape[1], exposure_3d.data.shape[2], rmf.data.shape[1], exposure_3d.data.shape[0]]
    source_model = CombinedModel3DIntConvolveEdisp(dimensions=dimensions, use_psf=True, exposure=exposure_3d,
                                                   psf=psf_3d,
                                                   spatial_model=spatial_model, spectral_model=spectral_model,
                                                   edisp=rmf.data)

    # Set starting values
    center = SkyCoord.from_name("Crab").galactic
    source_model.gamma = 2.2
    source_model.xpos = center.l.value
    source_model.ypos = center.b.value
    source_model.fwhm = 0.12
    source_model.ampl = 1.0

    # Fit
    model = bkg + 1E-11 * (source_model)
    fit = Fit(data=cube, model=model, stat=Cash(), method=NelderMead(), estmethod=Covariance())
    result = fit.fit()

    # TODO: The fact that it doesn't converge to the right Crab postion, flux and source size is due to the dummy psf
    reference = [183.73086704623555,
                 -11.426439339270253,
                 35843.352273899836,
                 3.7381063514549009,
                 2.2288775690653875]

    assert_allclose(result.parvals, reference, rtol=1E-5)
