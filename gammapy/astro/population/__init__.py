# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Astrophysical population models."""

from .simulate import (
    make_catalog_random_positions_cube,
    make_catalog_random_positions_sphere,
    make_base_catalog_galactic,
    add_snr_parameters,
    add_pulsar_parameters,
    add_pwn_parameters,
    add_observed_parameters,
)

from .spatial import (
    CaseBattacharya1998,
    FaucherKaspi2006,
    Lorimer2006,
    Paczynski1990,
    YusifovKucuk2004,
    YusifovKucuk2004B,
    Exponential,
    LogSpiral,
    FaucherSpiral,
    ValleeSpiral,
    radial_distributions,
)
from .velocity import (
    FaucherKaspi2006VelocityMaxwellian,
    FaucherKaspi2006VelocityBimodal,
    Paczynski1990Velocity,
    velocity_distributions,
)


__all__ = [
    "make_catalog_random_positions_cube",
    "make_catalog_random_positions_sphere",
    "make_base_catalog_galactic",
    "add_snr_parameters",
    "add_pulsar_parameters",
    "add_pwn_parameters",
    "add_observed_parameters",
    "CaseBattacharya1998",
    "FaucherKaspi2006",
    "Lorimer2006",
    "Paczynski1990",
    "YusifovKucuk2004",
    "YusifovKucuk2004B",
    "Exponential",
    "LogSpiral",
    "FaucherSpiral",
    "ValleeSpiral",
    "radial_distributions",
    "FaucherKaspi2006VelocityMaxwellian",
    "FaucherKaspi2006VelocityBimodal",
    "Paczynski1990Velocity",
    "velocity_distributions",
]
