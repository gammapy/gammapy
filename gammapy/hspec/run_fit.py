#! /usr/bin/env python

""" Reads a set of pha files and performs a maximum likelihood fit.
"""

import argparse as ap
parser = ap.ArgumentParser() 
parser.add_argument('InputFiles', metavar='InputFiles', type=str, nargs='+',\
                        help='Input pha files, typically run?????.pha, with the background, RMF and ARF defined in their header (any wildcard accepted)')
parser.add_argument('Model', metavar='Model', type=str, nargs='+',\
                        help='Sherpa model(s) to be fitted. Pre-setup models: PowLaw1D, logparabola, plexpcutoff; or multiple components. Example: PowLaw1D+logparabola')
parser.add_argument('--noplot', action='store_true',\
                        help='Do not produce the spectral plot (no graphical output).')
parser.add_argument('--manual', action='store_true',\
                        help='Set initial values and freeze parameters manually. This is default in case a non pre-setup model is called.')
parser.add_argument('--reproj',type=int,default=3,help='Reprojection level to use 1 to 3, default is 3. level 0 is no reprojection at all')
parser.add_argument('--conf', action='store_true',\
                        help='Compute confidence values for the fitted parameters')

args = parser.parse_args()

# Remaining imports
from sherpa.astro.ui import * # TEST if needed (only when outside sherpa env)
# Own packages
import logging
import load_model
import make_plot
from specsource import *

logger = logging.getLogger("sherpa")
logger.setLevel(logging.ERROR)

# Read and load the data and model:
listfile = args.InputFiles 
list_data = []

emax = 9e10 # Maximum energy taken into account for the fit --> Compute properly (as a function of the max E event?)
set_conf_opt("max_rstat",10000)  # We don't use a specific maximum reduced statistic value since we don't expect the cstat to be anywhere near the large number limit

p1 = load_model.load_model(args.Model[0]) #[-1] #load model returns an array of model components
                                          # where the last component is the total model
spec = SpecSource('SRC',listfile)

if args.reproj == 0:
    myspec = spec
elif args.reproj == 3:
    myspec = spec.reproject(nbins={'offset':5,'eff':10,'zen':10})
elif args.reproj == 2:
    myspec = spec.reproject(nbins={'offset':5,'eff':20,'zen':12})
elif args.reproj == 1:
    myspec = spec.reproject(nbins={'offset':25,'eff':40,'zen':30})
elif args.reproj == 4:
    myspec = spec.reproject(nbins={'offset':1,'eff':1,'zen':1})
    
myspec.set_source(p1)

if args.manual:
    load_model.set_manual_model(get_model(datid)) # Parameters for all runs are linked
else:
    print 'Using default initial values for model parameters'

myspec.fit(do_conf=args.conf)

if args.noplot:
    quit()
make_plot.make_plot(list_data,p1)

raw_input('Press <ENTER> to continue')
