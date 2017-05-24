import yaml
import numpy as np
import astropy.units as u
from gammapy.cube import SkyCube
from gammapy.extern.pathlib import Path
from astropy.coordinates import SkyCoord, Angle
from examples.configuration import get_model, make_ref_cube

from sherpa.estmethods import Covariance
from sherpa.optmethods import NelderMead
from sherpa.stats import Cash
from sherpa.fit import Fit

from gammapy.cube.sherpa_ import (
    CombinedModel3DInt,
    CombinedModel3DIntConvolveEdisp,
    NormGauss2DInt,
)

from sherpa.models import PowLaw1D, TableModel
def load_cubes(config):
    #Load the different cubes needed for the analysis
    #counts cube
    cube_dir = Path(config['logging']['working_dir'])
    npred_cube = SkyCube.read(cube_dir / 'npred_cube.fits.gz')
    #print(npred_cube)
    #print(npred_cube.info())
    #exposure cube
    exposure_cube = SkyCube.read(cube_dir / 'exposure_cube.fits', format='fermi-exposure')
    print(exposure_cube)
    print('exposure sum: {}'.format(np.nansum(exposure_cube.data)))
    i_nan = np.where(np.isnan(exposure_cube.data))
    exposure_cube.data[i_nan] = 0

    npred_cube_convolved = SkyCube.read(cube_dir / 'npred_cube_convolved.fits.gz')

    return dict(counts=npred_cube, exposure=exposure_cube)



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
    ref_cube = make_ref_cube(config)
    target_position = SkyCoord(config['model']['ra1'], config['model']['dec1'], unit="deg").galactic

    cubes = load_cubes(config)
    print('which available cubes:', cubes)
    # Define a 2D gaussian for the spatial model
    spatial_model = NormGauss2DInt('spatial-model')

    # Define a power law for the spectral model
    spectral_model = PowLaw1D('spectral-model')

    coord = cubes['counts'].sky_image_ref.coordinates(mode="edges")
    energies = cubes['counts'].energies(mode='edges').to("TeV")
    print('my energy bins: ', energies)
    #import IPython; IPython.embed();

    source_model = get_model(config)
    source_model_cube = source_model.evaluate_cube(ref_cube)
    model = source_model_cube #+ background_model

    print('------------------------------------')
    print('model ', model)
    print('------------------------------------ fitting')
    fit = Fit(
        data=cubes['counts'],
        model=model,
        stat=Cash(),
        method=NelderMead(),
        estmethod=Covariance(),
    )
    fit_results = fit.fit()
    print(fit_results.format())

    print('------------------------------------ end fitting')
    source_model2 = CombinedModel3DInt(
        coord=coord,
        energies=energies,
        use_psf=True,
        exposure=cubes['exposure'],
        psf=psf_3d,
        spatial_model=spatial_model,
        spectral_model=spectral_model,
    )

if __name__ == '__main__':
    main()
