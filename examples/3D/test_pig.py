import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from gammapy.data import DataStore
from gammapy.cube.pig_002 import MakeMaps
from gammapy.maps import WcsGeom,MapAxis

axis=MapAxis.from_edges(np.logspace(-1.,1.,10),unit=u.TeV)
geom=WcsGeom.create(skydir=(0,0),binsz=0.02,width=15.,coordsys='GAL',axes=[axis])

maker=MakeMaps(geom,4.*u.deg)
ds = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps/')

for obsid in ds.obs_table['OBS_ID']:
    obs=ds.obs(obsid)
    maker(obs)

