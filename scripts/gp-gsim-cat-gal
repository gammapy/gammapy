#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Simulate a catalog of Galactic sources.

Several spatial and velocity distributions are available
and each source has associated PSR, PWN und SNR parameters.
"""

# Parse command line arguments

from gammapy.utils.scripts import argparse, GammapyFormatter
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=GammapyFormatter)
parser.add_argument('outfile', type=str,
                    help='Output filename')
parser.add_argument('nsources', type=int,
                    help='Number of sources to simulate')
parser.add_argument('--max_age', type=float, default=1e6,
                    help='Simulation time interval')
parser.add_argument('--n_ISM', type=float, default=1,
                    help='Interstellar medium density')
parser.add_argument('--E_SN', type=float, default=1e51,
                    help='SNR kinetic energy')
parser.add_argument('--clobber', action='store_true',
                    help='Clobber output files?')
args = parser.parse_args()
print(args)
#args = vars(args)

# Execute script

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
from gammapy.astro.population import simulate
from gammapy.astro.population import spatial, velocity 

# TODO: Make rad_dis and vel_dis string options

# Draw random positions and velocities 
table = simulate.make_cat_gal(args.nsources,
                              rad_dis=spatial.YK04, 
                              vel_dis=velocity.H05, 
                              max_age=args.max_age,
                              n_ISM=args.n_ISM)

# Add intrinsic and observable source properties
table = simulate.add_par_snr(table, E_SN=args.E_SN)
table = simulate.add_par_psr(table)
table = simulate.add_par_pwn(table)
table = simulate.add_par_obs(table)

# TODO: store_options(table, options)
table.write(args.outfile, overwrite=args.clobber)
