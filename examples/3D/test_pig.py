"""Example 3D analysis using gammapy.maps.

Work in progress ...
"""
import numpy as np
import astropy.units as u
from gammapy.data import DataStore
from gammapy.maps import WcsGeom, MapAxis
from gammapy.cube import MapMaker

axis = MapAxis.from_edges(np.logspace(-1., 1., 10), unit=u.TeV)
geom = WcsGeom.create(skydir=(0, 0), binsz=0.02, width=15., coordsys='GAL', axes=[axis])

maker = MapMaker(geom, 4. * u.deg)
ds = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps/')

for obsid in ds.obs_table['OBS_ID']:
    obs = ds.obs(obsid)
    maker.process_obs(obs)

# TODO: check results
