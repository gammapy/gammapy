"""Example how to simulate and fit a 3D map using CTA IRFs.

"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from gammapy.irf import EffectiveAreaTable2D, EnergyDispersion2D, EnergyDependentMultiGaussPSF, Background3D
from gammapy.maps import WcsGeom, MapAxis, Map, WcsNDMap
from gammapy.spectrum.models import PowerLaw
from gammapy.image.models import SkyGaussian2D
from gammapy.cube import make_map_exposure_true_energy
from gammapy.cube import SkyModel, SkyModelMapEvaluator


def get_irfs():
    filename = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
    aeff = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')
    edisp = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')
    bkg = Background3D.read(filename, hdu='BACKGROUND')
    return dict(psf=psf, aeff=aeff, edisp=edisp, bkg=bkg)


def get_sky_model():
    spatial_model = SkyGaussian2D(
        lon_0='0.2 deg',
        lat_0='0.1 deg',
        sigma='0.2 deg',
    )
    spectral_model = PowerLaw(
        index=3,
        amplitude='1e-11 cm-2 s-1 TeV-1',
        reference='1 TeV',
    )
    return SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
    )


def get_geom():
    axis = MapAxis.from_edges(np.logspace(-1., 1., 10), unit=u.TeV)
    return WcsGeom.create(skydir=(0, 0), binsz=0.02, width=(8, 3), coordsys='GAL', axes=[axis])


def main():
    # Define some parameters
    pointing = SkyCoord(1, 0.5, unit='deg', frame='galactic')
    livetime = 1 * u.hour
    offset_max = 3 * u.deg
    offset = Angle('2 deg')

    irfs = get_irfs()
    geom = get_geom()
    sky_model = get_sky_model()

    # Let's get started ...
    exposure_map = make_map_exposure_true_energy(
        pointing=pointing, livetime=livetime, aeff=irfs['aeff'],
        ref_geom=geom, offset_max=offset_max,
    )
    exposure_map.write('exposure.fits')

    evaluator = SkyModelMapEvaluator(sky_model, exposure_map)

    # Accessing and saving a lot of the following maps is for debugging.
    # Just for a simulation one doesn't need to store all these things.
    dnde = evaluator.compute_dnde()
    WcsNDMap(geom, dnde).write('dnde.fits')

    flux = evaluator.compute_flux()
    WcsNDMap(geom, flux).write('flux.fits')

    npred = evaluator.compute_npred()
    WcsNDMap(geom, npred).write('npred.fits')

    # TODO: Apply PSF convolution
    psf = irfs['psf'].to_energy_dependent_table_psf(theta=offset)

    # kernels = irfs['psf'].kernels(npred_cube_simple)
    # npred_cube_convolved = npred_cube_simple.convolve(kernels)

    # TODO: optionally apply EDISP
    edisp = irfs['edisp'].to_energy_dispersion(offset=offset)

    # TODO: add background

    # Compute counts as a Poisson fluctuation
    counts = np.random.poisson(npred)
    WcsNDMap(geom, counts).write('counts.fits')


if __name__ == '__main__':
    main()
