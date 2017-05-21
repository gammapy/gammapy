# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Cube analysis utility functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy import units as u
from ..utils.energy import EnergyBounds
from ..spectrum import LogEnergyAxis
from .core import SkyCube

__all__ = [
    'compute_npred_cube',
    'compute_npred_cube_simple',
]


def compute_npred_cube(flux_cube, exposure_cube, energy_bins,
                       integral_resolution=10):
    """Compute predicted counts cube.

    TODO: describe what's passed in.
    I think it's a surface brightness in e.g. 'cm-2 s-1 TeV-1 sr-1'

    Parameters
    ----------
    flux_cube : `SkyCube`
        Differential flux cube
    exposure_cube : `SkyCube`
        Exposure cube
    integral_resolution : int (optional)
        Number of integration steps in energy bin when computing integral flux.

    Returns
    -------
    npred_cube : `SkyCube`
        Predicted counts cube in energy bins.

    See also
    --------
    compute_npred_cube_simple

    Examples
    --------
    Load an example dataset::

        from gammapy.datasets import FermiGalacticCenter
        from gammapy.utils.energy import EnergyBounds
        from gammapy.irf import EnergyDependentTablePSF
        from gammapy.cube import SkyCube, compute_npred_cube
        
        filenames = FermiGalacticCenter.filenames()
        flux_cube = SkyCube.read(filenames['diffuse_model'], format='fermi-background')
        exposure_cube = SkyCube.read(filenames['exposure_cube'], format='fermi-exposure')
        psf = EnergyDependentTablePSF.read(filenames['psf'])
        
    Compute an ``npred`` cube and a PSF-convolved version::

        flux_cube = flux_cube.reproject(exposure_cube)
        energy_bounds = EnergyBounds([10, 30, 100, 500], 'GeV')
        npred_cube = compute_npred_cube(flux_cube, exposure_cube, energy_bounds)
        
        kernels = psf.kernels(npred_cube)
        npred_cube_convolved = npred_cube.convolve(kernels)
    """
    _validate_inputs(flux_cube, exposure_cube)

    energy_axis = LogEnergyAxis(energy_bins, mode='edges')
    wcs = exposure_cube.wcs.deepcopy()

    energy_centers = EnergyBounds(energy_bins).log_centers

    # TODO: find a nicer way to do the iteration: make an empty 3D cube, then fill slice by slice
    data = []
    for ecenter, emin, emax in zip(energy_centers, energy_bins[:-1], energy_bins[1:]):
        flux_int = flux_cube.sky_image_integral(emin, emax, interpolation='linear',
                                                nbins=integral_resolution)

        exposure = exposure_cube.sky_image(ecenter, interpolation='linear')
        solid_angle = exposure.solid_angle()
        npred = flux_int.data * exposure.data * solid_angle
        data.append(npred)

    data = u.Quantity(data, '')

    return SkyCube(data=data, wcs=wcs, energy_axis=energy_axis)


def compute_npred_cube_simple(flux_cube, exposure_cube):
    """Compute npred cube.

    Multiplies flux and exposure and pixel solid angle and energy bin width

    This function is over 10 times faster than the one above
    # and gives slightly different results!
    # TODO: merge the two functions (or expose a uniform API, colling into other functions). Add tests and benchmark a bit!


    TODO: remove this function and instead fix the one in Gammapy (and add tests there)!
    This function was only added here to debug `gammapy.cube.utils.compute_npred_cube`
    After debugging the results almost match (differences due to integration method in energy)

    The one remaining issue with the function in Gammapy is that it gives NaN where flux = 0
    This must have to do with the integration method in energy and should be fixed.

    See also
    --------
    compute_npred_cube
    """
    _validate_inputs(flux_cube, exposure_cube)

    solid_angle = exposure_cube.sky_image_ref.solid_angle()
    de = exposure_cube.energy_width
    flux = flux_cube.data
    exposure = exposure_cube.data
    npred = flux * exposure * solid_angle * de[:, np.newaxis, np.newaxis]

    npred_cube = SkyCube.empty_like(exposure_cube)
    npred_cube.data = npred.to('')
    return npred_cube


def _validate_inputs(flux_cube, exposure_cube):
    if flux_cube.data.shape[1:] != exposure_cube.data.shape[1:]:
        raise ValueError('flux_cube and exposure cube must have the same shape!\n'
                         'flux_cube: {0}\nexposure_cube: {1}'
                         ''.format(flux_cube.data.shape[1:], exposure_cube.data.shape[1:]))
