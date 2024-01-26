"""
Survey Map Script
=================

Make a survey counts map using a script.

We create an all-sky map in AIT projection for the
`H.E.S.S. DL3 DR1 <https://www.mpi-hd.mpg.de/hfm/HESS/pages/dl3-dr1/>`__
dataset.
"""
import logging
from gammapy.data import DataStore
from gammapy.maps import Map

log = logging.getLogger(__name__)


def main():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    obs_id = data_store.obs_table["OBS_ID"]
    observations = data_store.get_observations(obs_id)

    m = Map.create()
    for obs in observations:
        log.info(f"Processing obs_id: {obs.obs_id}")
        m.fill_events(obs.events)

    m.write("survey_map.fits.gz", overwrite=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
