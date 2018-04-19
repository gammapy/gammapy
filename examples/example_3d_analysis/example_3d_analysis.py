"""Example 3D analysis using gammapy.maps.
"""
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.data import DataStore
from gammapy.maps import WcsGeom, MapAxis
from gammapy.cube import MapMaker

axis = MapAxis.from_edges(np.logspace(-1., 1., 10), unit=u.TeV)
geom = WcsGeom.create(skydir=(0, 0), binsz=0.02, width=15., coordsys='GAL', axes=[axis])

maker = MapMaker(geom, 4. * u.deg)
ds = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps/')

# obs_ids = [110380, 111140, 111159]
# obs_ids = [110380, 111140]
obs_ids = [110380]

for obs_id in obs_ids:
    print(obs_id)
    obs = ds.obs(obs_id)
    maker.process_obs(obs)

filename = 'counts.fits'
print(f'Writing {filename}')
maker.count_map.to_hdulist().writeto(filename, overwrite=True)

filename = 'background.fits'
print(f'Writing {filename}')
maker.background_map.to_hdulist().writeto(filename, overwrite=True)

filename = 'exposure.fits'
print(f'Writing {filename}')
maker.exposure_map.to_hdulist().writeto(filename, overwrite=True)


# import IPython; IPython.embed()
assert_allclose(maker.count_map.data.sum(), 50936)
assert_allclose(maker.background_map.data.sum(), 50936.02)
assert_allclose(maker.exposure_map.data.mean(), 405107140.0)


# TODO: compute PSF kernel

# TODO: check results and use in 3D analysis
