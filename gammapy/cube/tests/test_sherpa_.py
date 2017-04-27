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


@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def test_sherpa_crab_fit():
    from sherpa.models import NormGauss2D, PowLaw1D, TableModel
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

    model = 1e-9 * exposure * source_model  # 1e-9 flux factor

    fit = Fit(data=cube, model=model, stat=Chi2ConstVar(), method=LevMar())
    result = fit.fit()

    reference = [0.121556,
                 83.625627,
                 22.015564,
                 0.096903,
                 2.240989]

    assert_allclose(result.parvals, reference, rtol=1e-3)


@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def testCombinedModel3DInt():
    from sherpa.models import PowLaw1D, TableModel
    from sherpa.estmethods import Covariance
    from sherpa.optmethods import NelderMead
    from sherpa.stats import Cash
    from sherpa.fit import Fit
    from ..sherpa_ import CombinedModel3DInt, NormGauss2DInt

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

    coord = counts_3d.sky_image_ref.coordinates(mode="edges")
    energies = counts_3d.energies(mode='edges').to("TeV")
    source_model = CombinedModel3DInt(
        coord=coord, energies=energies, use_psf=True, exposure=exposure_3d, psf=psf_3d,
        spatial_model=spatial_model, spectral_model=spectral_model,
    )

    # Set starting values
    center = SkyCoord(83.633083, 22.0145, unit="deg").galactic
    source_model.gamma = 2.2
    source_model.xpos = center.l.value
    source_model.ypos = center.b.value
    source_model.fwhm = 0.12
    source_model.ampl = 1.0

    # Fit
    model = bkg + 1e-11 * source_model
    fit = Fit(data=cube, model=model, stat=Cash(), method=NelderMead(), estmethod=Covariance())
    result = fit.fit()

    # TODO: The fact that it doesn't converge to the right Crab postion is due to the dummy psf
    reference = [1.841954e+02,
                 -6.169173e+00,
                 6.166281e+00,
                 7.656752e-02,
                 2.306480e+00]

    assert_allclose(result.parvals, reference, rtol=1e-3)

    # Add a region to exclude in the fit: Here we will exclude some events from the Crab since there is no region to
    # exclude in the FOV for this example. It's just an example to show how it works and how to proceed in the fit.
    # Read the mask for the exclude region
    filename_mask = gammapy_extra.filename('test_datasets/cube/mask.fits')
    cube_mask = SkyCube.read(filename_mask)
    index_region_selected_3d = np.where(cube_mask.data.value == 1)

    # Set the counts and create a gammapy Data3DInt object
    # on which we apply a mask for the region we don't want to use in the fit
    cube = counts_3d.to_sherpa_data3d(dstype='Data3DInt')
    cube.mask = cube_mask.data.value.ravel()

    # Set the bkg and select only the data points of the selected region
    bkg = TableModel('bkg')
    bkg.load(None, bkg_3d.data.value[index_region_selected_3d].ravel())
    bkg.ampl = 1
    bkg.ampl.freeze()

    # The model is evaluated on all the points then it is compared with the data only on the selected_region
    source_model = CombinedModel3DInt(
        coord=coord, energies=energies, use_psf=True, exposure=exposure_3d, psf=psf_3d,
        spatial_model=spatial_model, spectral_model=spectral_model, select_region=True,
        index_selected_region=index_region_selected_3d,
    )
    # Set starting values
    source_model.gamma = 2.2
    source_model.xpos = center.l.value
    source_model.ypos = center.b.value
    source_model.fwhm = 0.12
    source_model.ampl = 1.0

    # Fit
    model = bkg + 1e-11 * (source_model)
    fit = Fit(data=cube, model=model, stat=Cash(), method=NelderMead(), estmethod=Covariance())
    result2 = fit.fit()

    # TODO: The fact that it doesn't converge to the right Crab postion is due to the dummy psf
    reference2 = [1.842016e+02,
                  -6.159901e+00,
                  5.419152e+00,
                  8.658486e-02,
                  2.298557e+00]
    assert_allclose(result2.parvals, reference2, rtol=1e-3)


@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def testCombinedModel3DIntConvolveEdisp():
    from sherpa.models import PowLaw1D, TableModel
    from sherpa.estmethods import Covariance
    from sherpa.optmethods import NelderMead
    from sherpa.stats import Cash
    from sherpa.fit import Fit
    from ..sherpa_ import CombinedModel3DIntConvolveEdisp, NormGauss2DInt

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
    coord = counts_3d.sky_image_ref.coordinates(mode="edges")
    energies = counts_3d.energies(mode='edges').to("TeV")
    source_model = CombinedModel3DIntConvolveEdisp(
        coord=coord, energies=energies, use_psf=True, exposure=exposure_3d, psf=psf_3d,
        spatial_model=spatial_model, spectral_model=spectral_model, edisp=rmf.data.data,
    )

    # Set starting values
    center = SkyCoord(83.633083, 22.0145, unit="deg").galactic
    source_model.gamma = 2.2
    source_model.xpos = center.l.value
    source_model.ypos = center.b.value
    source_model.fwhm = 0.12
    source_model.ampl = 1.0

    # Fit
    model = bkg + 1e-11 * (source_model)
    fit = Fit(data=cube, model=model, stat=Cash(), method=NelderMead(), estmethod=Covariance())
    result = fit.fit()

    # TODO: The fact that it doesn't converge to the right Crab postion, flux and source size is due to the dummy psf
    reference = [1.841920e+02,
                 -6.175650e+00,
                 6.227756e+00,
                 7.129763e-02,
                 2.269026e+00]
    assert_allclose(result.parvals, reference, rtol=1e-3)

    # Add a region to exclude in the fit: Here we will exclude some events from the Crab since there is no region to
    # exclude in the FOV for this example. It's just an example to show how it works and how to proceed in the fit.
    # Read the mask for the exclude region
    filename_mask = gammapy_extra.filename('test_datasets/cube/mask.fits')
    cube_mask = SkyCube.read(filename_mask)
    index_region_selected_3d = np.where(cube_mask.data.value == 1)

    # Set the counts and create a gammapy Data3DInt object
    # on which we apply a mask for the region we don't want to use in the fit
    cube = counts_3d.to_sherpa_data3d(dstype='Data3DInt')
    cube.mask = cube_mask.data.value.ravel()

    # Set the bkg and select only the data points of the selected region
    bkg = TableModel('bkg')
    bkg.load(None, bkg_3d.data.value[index_region_selected_3d].ravel())
    bkg.ampl = 1
    bkg.ampl.freeze()

    # The model is evaluated on all the points then it is compared with the data only on the selected_region
    source_model = CombinedModel3DIntConvolveEdisp(
        coord=coord, energies=energies, use_psf=True, exposure=exposure_3d,
        psf=psf_3d, spatial_model=spatial_model,
        spectral_model=spectral_model,
        edisp=rmf.data.data, select_region=True,
        index_selected_region=index_region_selected_3d,
    )

    # Set starting values
    source_model.gamma = 2.2
    source_model.xpos = center.l.value
    source_model.ypos = center.b.value
    source_model.fwhm = 0.12
    source_model.ampl = 1.0

    # Fit
    model = bkg + 1e-11 * (source_model)
    fit = Fit(data=cube, model=model, stat=Cash(), method=NelderMead(), estmethod=Covariance())
    result2 = fit.fit()

    # TODO: The fact that it doesn't converge to the right Crab postion is due to the dummy psf
    reference2 = [1.841921e+02,
                  -6.169094e+00,
                  5.497368e+00,
                  7.513465e-02,
                  2.250965e+00]
    assert_allclose(result2.parvals, reference2, rtol=1e-3)
