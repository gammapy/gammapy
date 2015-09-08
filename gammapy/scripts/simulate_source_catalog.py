# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
from ..utils.scripts import get_parser
from ..astro import population

__all__ = ['simulate_source_catalog']

log = logging.getLogger(__name__)


def main(args=None):
    parser = get_parser(simulate_source_catalog)
    parser.add_argument('outfile', type=str,
                        help='Output filename')
    parser.add_argument('nsources', type=float,
                        help='Number of sources to simulate')
    parser.add_argument('--max_age', type=float, default=1e6,
                        help='Simulation time interval (yr)')
    parser.add_argument('--ism-density', type=float, default=1,
                        help='Interstellar medium density (cm^-3)')
    parser.add_argument('--supernova-energy', type=float, default=1e51,
                        help='SNR kinetic energy (erg)')
    parser.add_argument('--radial_distribution', type=str, default='YK04',
                        help='Galactic radial source distribution')
    parser.add_argument('--velocity_distribution', type=str, default='H05',
                        help='Source velocity distribution')
    parser.add_argument('--spiral-arms', action='store_true',
                        help='Put a spiral arm pattern')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    args = parser.parse_args(args)
    simulate_source_catalog(**vars(args))


def simulate_source_catalog(outfile,
                            nsources,
                            max_age,
                            ism_density,
                            supernova_energy,
                            radial_distribution,
                            velocity_distribution,
                            spiral_arms,
                            overwrite):
    """Simulate a catalog of Galactic sources.

    Several spatial and velocity distributions are available
    and each source has associated PSR, PWN und SNR parameters.
    """
    from gammapy.astro.population import simulate

    # TODO: Make rad_dis and vel_dis string options

    # Draw random positions and velocities
    table = simulate.make_cat_gal(int(nsources),
                                  rad_dis=radial_distribution,
                                  vel_dis=velocity_distribution,
                                  max_age=max_age,
                                  n_ISM=ism_density,
                                  spiralarms=spiral_arms)

    # Add intrinsic and observable source properties
    table = simulate.add_par_snr(table, E_SN=supernova_energy)
    table = simulate.add_par_psr(table)
    table = simulate.add_par_pwn(table)
    table = simulate.add_observed_parameters(table)
    table = simulate.add_par_obs(table)

    # TODO: store_options(table, options)
    log.info('Writing {}'.format(outfile))
    table.write(outfile, overwrite=overwrite)


def _make_list(distributions):
    ss = ""
    for name in distributions.keys():
        description = distributions[name].__doc__.splitlines()[0]
        ss += "{0:10s} : {1}\n".format(name, description)
    return ss


radial_distributions_list = _make_list(population.radial_distributions)
velocity_distributions_list = _make_list(population.velocity_distributions)

_doc_lists = """
Available radial distributions:
{}

Available velocity distributions:
{}

""".format(radial_distributions_list, velocity_distributions_list)

simulate_source_catalog.__doc__ += _doc_lists
