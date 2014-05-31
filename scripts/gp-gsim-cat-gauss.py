#!/usr/bin/env python
"""
Simulate a catalog of Gaussian sources.
Useful for testing the catalog pipeline.
"""

import cmdl
from catalogs.mc import make_cat_gauss_random
from utils.table import store_options

# ------------------------------------------------------------
# Parse command line options
# ------------------------------------------------------------
option_list = [cmdl.nsources]
option_list += [
cmdl.make_option("--glon_sigma",
        type="float", default=30,
        help="GLON distribution width [default=%default]"),
cmdl.make_option("--glat_sigma",
        type="float", default=1,
        help="GLAT distribution width [default=%default]"),
cmdl.make_option("--extension_mean",
        type="float", default=0.1,
        help="Extension distribution mean [default=%default]"),
cmdl.make_option("--extension_sigma",
        type="float", default=0.1,
        help="Extension distribution sigma [default=%default]"),
cmdl.make_option("--flux_index",
        type="float", default=1,
        help="Flux distribution power law index [default=%default]"),
cmdl.make_option("--flux_min",
        type="float", default=1e-11,
        help="Minimum flux [default=%default]"),
cmdl.make_option("--flux_max",
        type="float", default=1e-10,
        help="Maximum flux [default=%default]"),
]
option_list += [cmdl.clobber, cmdl.verbose]
argument_list = ['catalog']
options = cmdl.parse(option_list, argument_list)

# ------------------------------------------------------------
# Execute the program
# ------------------------------------------------------------
catalog = make_cat_gauss_random(**options.__dict__)
store_options(catalog, options)
catalog.write(options.catalog, overwrite=options.clobber)
