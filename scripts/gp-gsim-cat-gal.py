#!/usr/bin/env python
"""
Simulate a catalog of Galactic sources.

Several spatial and velocity distributions are available
and each source has associated PSR, PWN und SRN parameters.
"""
import cmdl
from utils.table import store_options
import catalogs.mc as sim
import distributions.spatial as spatial 
import distributions.velocity as velocity 

# ------------------------------------------------------------
# Parse command line options
# ------------------------------------------------------------
option_list = [cmdl.nsources]
option_list += [
cmdl.make_option("--max_age",
        type="float", default=1e6,
        help="Simulation time interval [default=%default]"),
cmdl.make_option("--n_ISM",
        type="float", default=1,
        help="Environment density [default=%default]"),
cmdl.make_option("--E_SN",
        type="float", default=1e51,
        help="SNR kinetic energy [default=%default]"),

]
option_list += [cmdl.clobber, cmdl.verbose]
argument_list = ['catalog']
options = cmdl.parse(option_list, argument_list)

# ------------------------------------------------------------
# Execute the program
# ------------------------------------------------------------

# TODO: Make rad_dis and vel_dis string options

# Draw random positions and velocities 
catalog = sim.make_cat_gal(options.nsources,
                           rad_dis=spatial.YK04, 
                           vel_dis=velocity.H05, 
                           max_age=options.max_age,
                           n_ISM=options.n_ISM)

# Add intrinsic and observable source properties
catalog = sim.add_par_snr(catalog, E_SN=options.E_SN)
catalog = sim.add_par_psr(catalog)
catalog = sim.add_par_pwn(catalog)
catalog = sim.add_par_obs(catalog)

# Write to file
store_options(catalog, options)
catalog.write(options.catalog, overwrite=options.clobber)
