# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Cube analysis utility functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy import units as u
from ..utils.energy import EnergyBounds
from ..spectrum import LogEnergyAxis
from .core import SkyCube

__all__ = [
    'compute_npred_cube',
]


def compute_npred_cube(flux_cube, exposure_cube, energy_bins,
                       integral_resolution=10):
    """Compute predicted counts cube in energy bins.

    Parameters
    ----------
    flux_cube : `SkyCube`
        Differential flux cube.
    exposure_cube : `SkyCube`
        Instrument exposure cube.
    integral_resolution : int (optional)
        Number of integration steps in energy bin when computing integral flux.

    Returns
    -------
    npred_cube : `SkyCube`
        Predicted counts cube in energy bins.
    """
    if flux_cube.data.shape[1:] != exposure_cube.data.shape[1:]:
        raise ValueError('flux_cube and exposure cube must have the same shape!\n'
                         'flux_cube: {0}\nexposure_cube: {1}'
                         ''.format(flux_cube.data.shape[1:], exposure_cube.data.shape[1:]))

    energy_axis = LogEnergyAxis(energy_bins, mode='edges')
    wcs = exposure_cube.wcs.deepcopy()

    energy_centers = EnergyBounds(energy_bins).log_centers

    data = []
    # TODO: find a nicer way to do the iteration
    for ecenter, emin, emax in zip(energy_centers, energy_bins[:-1], energy_bins[1:]):
        flux_int = flux_cube.sky_image_integral(emin, emax, interpolation='linear',
                                                nbins=integral_resolution)

        exposure = exposure_cube.sky_image(ecenter, interpolation='linear')
        npred = flux_int.data * exposure.data * exposure.solid_angle()
        data.append(npred)

    data = u.Quantity(data, '')

    return SkyCube(data=data, wcs=wcs, energy_axis=energy_axis)
