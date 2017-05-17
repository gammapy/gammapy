"""
Example how to simulate a 3D n_pred / n_obs cube.

Using CTA IRFs


Filename: ../../gammapy-extra/test_datasets/cta_1dc/caldb/data/cta/prod3b/bcf/South_z20_50h/irf_file.fits
No.    Name         Type      Cards   Dimensions   Format
  0  PRIMARY     PrimaryHDU       8   ()
  1  EFFECTIVE AREA  BinTableHDU     51   1R x 5C   [21E, 21E, 6E, 6E, 126E]
  2  POINT SPREAD FUNCTION  BinTableHDU     68   1R x 10C   [21E, 21E, 6E, 6E, 126E, 126E, 126E, 126E, 126E, 126E]
  3  ENERGY DISPERSION  BinTableHDU     54   1R x 7C   [60E, 60E, 300E, 300E, 6E, 6E, 108000E]
  4  BACKGROUND  BinTableHDU     57   1R x 7C   [12E, 12E, 12E, 12E, 21E, 21E, 3024E]

"""
import numpy as np
import astropy.units as u
import yaml
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from gammapy.cube import SkyCube, make_exposure_cube
from gammapy.irf import EnergyDependentMultiGaussPSF
from gammapy.irf import EffectiveAreaTable2D

from gammapy.spectrum.models import PowerLaw, ExponentialCutoffPowerLaw, LogParabola
from gammapy.image.models import Shell2D
import matplotlib.pyplot as plt

def get_irfs(config):
    filename = '$GAMMAPY_EXTRA/test_datasets/cta_1dc/caldb/data/cta/prod3b/bcf/South_z20_50h/irf_file.fits'

    psf_fov = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
    psf = psf_fov.to_energy_dependent_table_psf(theta=Angle(config['selection']['offset_fov']*u.deg))
    aeff = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')

    return dict(psf=psf, aeff=aeff)


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

class Model3D(object):
    def __init__(self, spatial_model, spectral_model):
        self.spatial_model = spatial_model
        self.spectral_model = spectral_model

    def evaluate(self, lon, lat, energy):
        a = self.spatial_model.evaluate(lon, lat)
        b = self.spectral_model.evaluate(energy)
        return a * b

    def evaluate_cube(self, cube):
        coords = cube.sky_image_ref.coordinates()
        import IPython;
        IPython.embed();
        lon = coords.data.lon
        lat = coords.data.lat
        energy = cube.energies(mode='bounds')
        # TODO: decide where to integrate over spatial and energy,
        # e.g. multiply with pixel solid angle and energy bin width
        return self.evaluate(lon, lat, energy)

def get_model():
    if (config['model']['template1'] == 'Shell2D'):
        spatial_model = Shell2D(
            amplitude=1,
            x_0=config['model']['ra1'],
            y_0=config['model']['dec1'],
            r_in=config['model']['rin1'],
            width=config['model']['width1'],
            normed=True, # not sure if we need this.
         )

    if (config['model']['spectrum1']=='PL'):
        spectral_model = PowerLaw(
                amplitude=config['model']['prefactor1']* u.Unit('cm-2 s-1 TeV-1'),
                index=config['model']['index1'],
                reference=config['model']['pivot_energy1']* u.Unit('TeV'),
        )
    if (config['model']['spectrum1']=='PLEC'):
        spectral_model = ExponentialCutoffPowerLaw(
                amplitude=config['model']['prefactor1']* u.Unit('cm-2 s-1 TeV-1'),
                index=config['model']['index1'],
                reference=config['model']['pivot_energy1']* u.Unit('TeV'),
                lambda_=config['model']['cutoff']* u.Unit('TeV-1'),
        )

    return Model3D(spatial_model, spectral_model)



if __name__ == '__main__':

    with open('config.yaml') as fh:
        config = yaml.load(fh)

    # getting the IRFs, effective area and PSF
    irfs = get_irfs(config)
    # create an empty reference image
    ref_cube = make_ref_cube(config)
    if (config['binning']['coordsys'] == 'CEL'):
        pointing = SkyCoord(config['pointing']['ra'], config['pointing']['dec'], frame='icrs', unit='deg')
    if (config['binning']['coordsys'] == 'GAL'):
        pointing = SkyCoord(config['pointing']['glat'], config['pointing']['glon'], frame='galactic', unit='deg')
    # compute the exposure map
    exposure = make_exposure_cube(
        pointing=pointing,
        livetime=config['pointing']['livetime'],
        aeff=irfs['aeff'],
        ref_cube=ref_cube,
        offset_max=Angle(config['selection']['ROI']*u.deg),
    )
    # exposure.show()

    model = get_model()

    flux_cube = model.evaluate_cube(ref_cube)
    # n_pred = compute_npred(model, exposure, psf)
    # n_obs = np.random.poisson(n_pred)
