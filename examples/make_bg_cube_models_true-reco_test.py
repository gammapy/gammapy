from __future__ import (absolute_import, division, print_function,
                        unicode_literals) # python 2 as python 3
import os
import numpy as np
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.time import Time
from gammapy.datasets import make_test_bg_cube_model, make_test_dataset
from gammapy.obs import (DataStore, ObservationGroups,
                         ObservationGroupAxis)
from gammapy.background import make_bg_cube_model


#TEST = True # create a small dataset
TEST = False # create a large dataset (slow)


# using values from scripts.group_observations
AZ_RANGE = Angle([90, 270], 'degree')
ALT_RANGE = Angle([72, 90], 'degree')
GROUP_ID = 27
OUTDIR = 'bg_cube_models'
OVERWRITE = False


def create_dummy_observation_grouping():
    """Define dummy observation grouping."""

    altitude_edges = ALT_RANGE
    azimuth_edges = AZ_RANGE
    group_id = GROUP_ID

    list_obs_group_axis = [ObservationGroupAxis('ALT',
                                                altitude_edges,
                                                'bin_edges'),
                           ObservationGroupAxis('AZ',
                                                azimuth_edges,
                                                'bin_edges')]
    obs_groups = ObservationGroups(list_obs_group_axis)
    obs_groups.obs_groups_table['GROUP_ID'][0] = group_id

    return obs_groups


def make_true_model():
    """Make a true bg cube model."""

    overwrite = OVERWRITE

    # create output folder
    outdir = OUTDIR + '/true'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    else:
        if overwrite:
            # delete and create again
            shutil.rmtree(outdir) # recursively
            os.mkdir(outdir)
        else:
            # do not overwrite, hence exit
            s_error = "Cannot continue: directory \'{}\' exists.".format(outdir)
            raise RuntimeError(s_error)

    # create dummy observation grouping
    obs_groups = create_dummy_observation_grouping()
    # save
    outfile = outdir + '/bg_observation_groups.ecsv'
    print('Writing {}'.format(outfile, overwrite=overwrite))
    obs_groups.write(outfile)

    # use binning from CubeBackgroundModel.define_cube_binning
    detx_range = (Angle(-0.07, 'radian').to('degree'),
                  Angle(0.07, 'radian').to('degree'))
    ndetx_bins = 60
    dety_range = (Angle(-0.07, 'radian').to('degree'),
                  Angle(0.07, 'radian').to('degree'))
    ndety_bins = 60
    energy_band = Quantity([0.1, 80.], 'TeV')
    nenergy_bins = 20

    # average altitude
    altitude = (ALT_RANGE[0]+ ALT_RANGE[1])/2.

    # using default from make_test_eventlist
    sigma = Angle(5., 'deg'),
    spectral_index = 2.7,

    bg_cube_model = make_test_bg_cube_model(detx_range=detx_range,
                                            ndetx_bins=ndetx_bins,
                                            dety_range=dety_range,
                                            ndety_bins=ndety_bins,
                                            energy_band=energy_band,
                                            nenergy_bins=nenergy_bins,
                                            altitude= altitude,
                                            apply_mask=False)

    # save
    group_id = GROUP_ID
    outfile = outdir + '/bg_cube_model_group{}'.format(group_id)
    print("Writing {}".format('{}_table.fits.gz'.format(outfile)))
    print("Writing {}".format('{}_image.fits.gz'.format(outfile)))
    bg_cube_model.write('{}_table.fits.gz'.format(outfile),
                        format='table', clobber=overwrite)
    bg_cube_model.write('{}_image.fits.gz'.format(outfile),
                        format='image', clobber=overwrite)


def make_reco_model():
    """Make a reco bg cube model."""

    SCHEME = 'HESS'
    METHOD = 'default'

    fits_path = 'test_dataset'
    overwrite = OVERWRITE
    test = TEST
    group_id = GROUP_ID

    # create output folder
    outdir = OUTDIR + '/reco'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    else:
        if overwrite:
            # delete and create again
            shutil.rmtree(outdir) # recursively
            os.mkdir(outdir)
        else:
            # do not overwrite, hence exit
            s_error = "Cannot continue: directory \'{}\' exists.".format(outdir)
            raise RuntimeError(s_error)

    # 0. create dummy observation grouping
    obs_groups = create_dummy_observation_grouping()
    # save
    outfile = outdir + '/bg_observation_groups.ecsv'
    print('Writing {}'.format(outfile, overwrite=overwrite))
    obs_groups.write(outfile)

    # 1. create dummy dataset

    observatory_name = SCHEME

    # use enough stats so that rebinning (and resmoothing?) doesn't take place
    n_obs = 100
    if test:
        # run fast
        n_obs = 2

    az_range = AZ_RANGE
    alt_range = ALT_RANGE
    random_state = np.random.RandomState(seed=0)

    make_test_dataset(fits_path=fits_path, overwrite=overwrite,
                      observatory_name='HESS', n_obs=n_obs,
                      az_range=az_range,
                      alt_range=alt_range,
                      date_range=(Time('2010-01-01T00:00:00',
                                       format='isot', scale='utc'),
                                  Time('2015-01-01T00:00:00',
                                       format='isot', scale='utc')),
                      n_tels_range=(3, 4),
                      random_state=random_state)

    # 2. get observation table
    scheme = SCHEME
    data_store = DataStore(dir=fits_path, scheme=scheme)
    observation_table = data_store.make_observation_table()

    # 3. build bg model
    outfile = outdir + '/bg_observation_table_group{}.fits.gz'.format(group_id)
    print("Writing {}".format(outfile))
    observation_table.write(outfile, overwrite=overwrite)

    method = METHOD
    bg_cube_model = make_bg_cube_model(observation_table=observation_table,
                                       fits_path=fits_path,
                                       method=method)

    # save
    outfile = outdir + '/bg_cube_model_group{}'.format(group_id)
    print("Writing {}".format('{}_table.fits.gz'.format(outfile)))
    print("Writing {}".format('{}_image.fits.gz'.format(outfile)))
    bg_cube_model.write('{}_table.fits.gz'.format(outfile),
                        format='table', clobber=overwrite)
    bg_cube_model.write('{}_image.fits.gz'.format(outfile),
                        format='image', clobber=overwrite)


if __name__ == '__main__':

    outdir = OUTDIR
    overwrite = OVERWRITE

    # create output folder
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    else:
        if overwrite:
            # delete and create again
            shutil.rmtree(outdir) # recursively
            os.mkdir(outdir)
        else:
            # do not overwrite, hence exit
            s_error = "Cannot continue: directory \'{}\' exists.".format(outdir)
            raise RuntimeError(s_error)

    make_true_model()
    make_reco_model()
    # use plot_bg_cube_model_comparison.py for comparing the models
