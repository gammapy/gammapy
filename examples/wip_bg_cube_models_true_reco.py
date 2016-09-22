"""
Script to produce true and reco background cube models.

The script uses the same model (except for absolute normalization) to
produce a true cube bg model and a reco cube bg model. The models can
be used to test the cube bg model production and can be compared to
each other using the plot_bg_cube_model_comparison.py example script.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.time import Time
from gammapy.extern.pathlib import Path
from gammapy.datasets import make_test_bg_cube_model, make_test_dataset
from gammapy.data import (DataStore, ObservationGroups,
                         ObservationGroupAxis)
from gammapy.background import make_bg_cube_model

# TEST = True # create a small dataset
TEST = False  # create a large dataset (slow)

# define model parameters
SIGMA = Angle(5., 'deg')
# INDEX = 2.7
# INDEX = 2.0
INDEX = 1.5

# define obs group bin
# using half hard-coded values from scripts.group_observations
AZ_RANGE = Angle([90, 270], 'degree')
ALT_RANGE = Angle([72, 90], 'degree')
GROUP_ID = 27

OUTDIR = 'bg_cube_models'
OVERWRITE = False


def make_bg_cube_models_true_reco():
    """Produce true and reco background cube models and store them to file.
    """
    outdir = OUTDIR
    overwrite = OVERWRITE

    # create output folder
    _create_dir(outdir, overwrite)

    make_true_model()
    make_reco_model()


def create_dummy_observation_grouping():
    """Define dummy observation grouping.

    Define an observation grouping with only one group.

    Returns
    -------
    obs_groups : `~gammapy.data.ObservationGroups`
        Observation grouping.
    """
    alt_axis = ObservationGroupAxis('ALT', ALT_RANGE, fmt='edges')
    az_axis = ObservationGroupAxis('AZ', AZ_RANGE, fmt='edges')
    obs_groups = ObservationGroups([alt_axis, az_axis])
    obs_groups.obs_groups_table['GROUP_ID'][0] = GROUP_ID

    return obs_groups


def make_true_model():
    """Make a true bg cube model."""

    overwrite = OVERWRITE

    # create output folder
    outdir = os.path.join(OUTDIR, 'true')
    _create_dir(outdir, overwrite)

    # create dummy observation grouping
    obs_groups = create_dummy_observation_grouping()
    # save
    outfile = os.path.join(outdir, 'bg_observation_groups.ecsv')
    print('Writing {}'.format(outfile, overwrite=overwrite))
    obs_groups.write(outfile)

    # use binning from FOVCubeBackgroundModel.define_cube_binning
    det_range = (Angle(-0.07, 'radian').to('degree'),
                 Angle(0.07, 'radian').to('degree'))
    ndet_bins = 60
    energy_band = Quantity([0.1, 80.], 'TeV')
    nenergy_bins = 20

    # average altitude
    altitude = (ALT_RANGE[0] + ALT_RANGE[1]) / 2.

    sigma = SIGMA
    spectral_index = INDEX

    bg_cube_model = make_test_bg_cube_model(detx_range=det_range,
                                            ndetx_bins=ndet_bins,
                                            dety_range=det_range,
                                            ndety_bins=ndet_bins,
                                            energy_band=energy_band,
                                            nenergy_bins=nenergy_bins,
                                            altitude=altitude,
                                            sigma=sigma,
                                            spectral_index=spectral_index,
                                            apply_mask=False,
                                            do_not_force_mev_units=True)

    # save
    group_id = GROUP_ID
    outfile = os.path.join(outdir, 'bg_cube_model_group{}'.format(group_id))
    print("Writing {}".format('{}_table.fits.gz'.format(outfile)))
    print("Writing {}".format('{}_image.fits.gz'.format(outfile)))
    bg_cube_model.write('{}_table.fits.gz'.format(outfile),
                        format='table', clobber=overwrite)
    bg_cube_model.write('{}_image.fits.gz'.format(outfile),
                        format='image', clobber=overwrite)


def make_reco_model():
    """Make a reco bg cube model."""

    METHOD = 'default'

    data_dir = 'test_dataset'
    overwrite = OVERWRITE
    test = TEST
    group_id = GROUP_ID

    # create output folder
    outdir = os.path.join(OUTDIR, 'reco')
    _create_dir(outdir, overwrite)

    # 0. create dummy observation grouping
    obs_groups = create_dummy_observation_grouping()
    # save
    outfile = os.path.join(outdir, 'bg_observation_groups.ecsv')
    print('Writing {}'.format(outfile, overwrite=overwrite))
    obs_groups.write(outfile)

    # 1. create dummy dataset

    # use enough stats so that rebinning (and resmoothing?) doesn't take place
    n_obs = 100
    if test:
        # run fast
        n_obs = 2

    az_range = AZ_RANGE
    alt_range = ALT_RANGE
    random_state = np.random.RandomState(seed=0)

    sigma = SIGMA
    spectral_index = INDEX

    make_test_dataset(outdir=data_dir, overwrite=overwrite,
                      observatory_name='HESS', n_obs=n_obs,
                      az_range=az_range,
                      alt_range=alt_range,
                      date_range=(Time('2010-01-01T00:00:00',
                                       format='isot', scale='utc'),
                                  Time('2015-01-01T00:00:00',
                                       format='isot', scale='utc')),
                      n_tels_range=(3, 4),
                      sigma=sigma,
                      spectral_index=spectral_index,
                      random_state=random_state)

    # 2. get observation table
    data_store = DataStore.from_dir(dir=data_dir)
    observation_table = data_store.make_observation_table()
    outfile = os.path.join(outdir, 'bg_observation_table_group{}.fits.gz'.format(group_id))
    print("Writing {}".format(outfile))
    observation_table.write(outfile, overwrite=overwrite)

    # 3. build bg model
    method = METHOD
    bg_cube_model = make_bg_cube_model(observation_table=observation_table,
                                       data_dir=data_dir,
                                       method=method,
                                       do_not_force_mev_units=True)

    # save
    outfile = os.path.join(outdir, 'bg_cube_model_group{}'.format(group_id))
    print("Writing {}".format('{}_table.fits.gz'.format(outfile)))
    print("Writing {}".format('{}_image.fits.gz'.format(outfile)))
    bg_cube_model.write('{}_table.fits.gz'.format(outfile),
                        format='table', clobber=overwrite)
    bg_cube_model.write('{}_image.fits.gz'.format(outfile),
                        format='image', clobber=overwrite)


if __name__ == '__main__':
    """Main function: launch the whole analysis chain.
    """
    make_bg_cube_models_true_reco()
