from gammapy.cube import SkyCube, CombinedModel3D
from gammapy.spectrum.models import PowerLaw, ExponentialCutoffPowerLaw
from gammapy.image.models import Shell2D, Sphere2D
import astropy.units as u

__all__ = [
    'get_model_gammapy',
    'make_ref_cube',
]


def get_model_gammapy(config):
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
    if config['model']['template1'] == 'Sphere2D':
        spatial_model = Sphere2D(
            amplitude=1,
            x_0=config['model']['ra1'],
            y_0=config['model']['dec1'],
            r_0=config['model']['rad'],
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

    return CombinedModel3D(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
    )


def make_ref_cube(config):
    WCS_SPEC = {
        'nxpix': config['binning']['nxpix'],
        'nypix': config['binning']['nypix'],
        'binsz': config['binning']['binsz'],
        'xref': config['pointing']['ra'],
        'yref': config['pointing']['dec'],
        'proj': config['binning']['proj'],
        'coordsys': config['binning']['coordsys'],
    }

    # define reconstructed energy binning
    ENERGY_SPEC = {
        'mode': 'edges',
        'enumbins': config['binning']['enumbins'],
        'emin': config['selection']['emin'],
        'emax': config['selection']['emax'],
        'eunit': 'TeV',
    }

    return SkyCube.empty(**WCS_SPEC, **ENERGY_SPEC)
