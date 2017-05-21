"""
An example how to simulate a 3D n_pred / n_obs cube.

Using CTA IRFs.

TODOs:
* For `compute_npred_cube` we're getting ``NaN`` values where flux == 0.
  This shouldn't happen and is bad. Figure out what's going on and fix!

"""
import numpy as np
import astropy.units as u
import yaml
from astropy.coordinates import SkyCoord, Angle
from gammapy.irf import EnergyDependentMultiGaussPSF
from gammapy.irf import EffectiveAreaTable2D, EnergyDispersion2D
from gammapy.spectrum.models import PowerLaw, ExponentialCutoffPowerLaw
from gammapy.image.models import Shell2D
from gammapy.cube import SkyCube, CombinedModel3D
from gammapy.cube import make_exposure_cube
from gammapy.cube.utils import compute_npred_cube, compute_npred_cube_simple


def get_irfs(config):
    filename = '$GAMMAPY_EXTRA/test_datasets/cta_1dc/caldb/data/cta/prod3b/bcf/South_z20_50h/irf_file.fits'

    offset = Angle(config['selection']['offset_fov'] * u.deg)

    psf_fov = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
    psf = psf_fov.to_energy_dependent_table_psf(theta=offset)

    aeff = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')

    edisp_fov = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')
    edisp = edisp_fov.to_energy_dispersion(offset=offset)

    # TODO: read background once it's working!
    # bkg = Background3D.read(filename, hdu='BACKGROUND')

    return dict(psf=psf, aeff=aeff, edisp=edisp)


def make_ref_cube(config):
    WCS_SPEC = {'nxpix': config['binning']['nxpix'],
                'nypix': config['binning']['nypix'],
                'binsz': config['binning']['binsz'],
                'xref': config['pointing']['ra'],
                'yref': config['pointing']['dec'],
                'proj': config['binning']['proj'],
                'coordsys': config['binning']['coordsys']}

    # define reconstructed energy binning
    ENERGY_SPEC = {'mode': 'edges',
                   'enumbins': config['binning']['enumbins'],
                   'emin': config['selection']['emin'],
                   'emax': config['selection']['emax'],
                   'eunit': 'TeV'}

    return SkyCube.empty(**WCS_SPEC, **ENERGY_SPEC)


def get_model(config):
    if config['model']['template1'] == 'Shell2D':
        spatial_model = Shell2D(
            amplitude=1,
            x_0=config['model']['ra1'],
            y_0=config['model']['dec1'],
            r_in=config['model']['rin1'],
            width=config['model']['width1'],
            # Note: for now we need spatial models that are normalised
            # to integrate to 1 or results will be incorrect!!!
            normed=True,
        )

    if config['model']['spectrum1'] == 'pl':
        spectral_model = PowerLaw(
            amplitude=config['model']['prefactor1'] * u.Unit('cm-2 s-1 TeV-1'),
            index=config['model']['index1'],
            reference=config['model']['pivot_energy1'] * u.Unit('TeV'),
        )
    if config['model']['spectrum1'] == 'ecpl':
        spectral_model = ExponentialCutoffPowerLaw(
            amplitude=config['model']['prefactor1'] * u.Unit('cm-2 s-1 TeV-1'),
            index=config['model']['index1'],
            reference=config['model']['pivot_energy1'] * u.Unit('TeV'),
            lambda_=config['model']['cutoff'] * u.Unit('TeV-1'),
        )

    return CombinedModel3D(spatial_model, spectral_model)


def compute_spatial_model_integral(model, image):
    """
    This is just used for debugging here.
    TODO: remove or put somewhere into Gammapy as a utility function or method?
    """
    coords = image.coordinates()
    surface_brightness = model(coords.data.lon.deg, coords.data.lat.deg) * u.Unit('deg-2')
    solid_angle = image.solid_angle()
    return (surface_brightness * solid_angle).sum().to('')


def read_config(filename):
    with open(filename) as fh:
        config = yaml.load(fh)

    # TODO: fix the following issue in a better way (e.g. raise an error or fix somehow)
    # apparently this gets returned as string, but we want float!?
    # prefactor1: 2e-12
    config['model']['prefactor1'] = float(config['model']['prefactor1'])

    return config


def main():
    config = read_config('config.yaml')
    # getting the IRFs, effective area and PSF
    irfs = get_irfs(config)
    # create an empty reference image
    ref_cube = make_ref_cube(config)
    if config['binning']['coordsys'] == 'CEL':
        pointing = SkyCoord(config['pointing']['ra'], config['pointing']['dec'], frame='icrs', unit='deg')
    if config['binning']['coordsys'] == 'GAL':
        pointing = SkyCoord(config['pointing']['glat'], config['pointing']['glon'], frame='galactic', unit='deg')

    livetime = u.Quantity(config['pointing']['livetime']).to('second')
    exposure_cube = make_exposure_cube(
        pointing=pointing,
        livetime=livetime,
        aeff=irfs['aeff'],
        ref_cube=ref_cube,
        offset_max=Angle(config['selection']['ROI']),
    )
    # exposure_cube.data = exposure_cube.data.to('m2 s')

    # Define model and do some quick checks
    model = get_model(config)
    spatial_integral = compute_spatial_model_integral(model.spatial_model, exposure_cube.sky_image_ref)
    print('Spatial integral (should be 1): ', spatial_integral)
    flux_integral = model.spectral_model.integral(emin='1 TeV', emax='10 TeV')
    print('Integral flux in range 1 to 10 TeV: ', flux_integral.to('cm-2 s-1'))
    # import IPython; IPython.embed()

    # Compute PSF-convolved npred in a few steps
    # 1. flux cube
    # 2. npred_cube
    # 3. apply PSF
    # 4. apply EDISP

    flux_cube = model.evaluate_cube(ref_cube)

    from time import time
    t0 = time()
    npred_cube = compute_npred_cube(
        flux_cube, exposure_cube,
        ebounds=flux_cube.energies('edges'),
        integral_resolution=2,
    )
    t1 = time()
    npred_cube_simple = compute_npred_cube_simple(flux_cube, exposure_cube)
    t2 = time()
    print('npred_cube: ', t1 - t0)
    print('npred_cube_simple: ', t2 - t1)
    print('npred_cube sum: {}'.format(np.nansum(npred_cube.data.to('').data)))
    print('npred_cube_simple sum: {}'.format(np.nansum(npred_cube_simple.data.to('').data)))

    # TODO: apply PSF convolution here
    # kernels = irfs['psf'].kernels(npred_cube)
    # npred_cube_convolved = npred_cube.convolve(kernels)

    # TODO: apply EDISP here!

    # Compute counts as a Poisson fluctuation
    # counts_cube = SkyCube.empty_like(ref_cube)
    # counts_cube.data = np.random.poisson(npred_cube_convolved.data)

    # Debugging output

    print('npred_cube sum: {}'.format(np.nansum(npred_cube.data.to('').data)))
    # TODO: check that sum after PSF convolution or applying EDISP are the same

    exposure_cube.write('exposure_cube.fits.gz', overwrite=True)
    flux_cube.write('flux_cube.fits.gz', overwrite=True)
    npred_cube.write('npred_cube.fits.gz', overwrite=True)
    # npred_cube_convolved.write('npred_cube_convolved.fits', overwrite=True)


if __name__ == '__main__':
    main()
