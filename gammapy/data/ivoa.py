# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import os
import numpy as np
from astropy.table import Column, Table
from gammapy.data import DataStore
from gammapy.utils.scripts import make_path

__all__ = ["to_obscore_table"]


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _obscore_def():
    """Generate the Obscore default table
    In case the obscore standard changes, this function should be changed according to https://www.ivoa.net/documents/ObsCore

    Returns
    -------
    table : `~astropy.table.Table`
       the empty table
    """
    n_obscore_val = 29  # Number of obscore values
    obscore_default = np.empty(n_obscore_val, dtype=object)
    obscore_default[0] = Column(
        name="dataproduct_type",
        unit="",
        description="Data product (file content) primary type",
        dtype="U10",
        meta={"Utype": "ObsDataset.dataProductType"},
    )
    obscore_default[1] = Column(
        name="calib_level",
        unit="",
        description="Calibration level of the observation: in {0, 1, 2, 3, 4}",
        dtype="i4",
        meta={"Utype": "ObsDataset.calibLevel"},
    )
    obscore_default[2] = Column(
        name="target_name",
        unit="",
        description="Object of interest",
        dtype="U25",
        meta={"Utype": "Target.name"},
    )
    obscore_default[3] = Column(
        name="obs_id",
        unit="",
        description="Internal ID given by the ObsTAP service",
        dtype="U10",
        meta={"Utype": "DataID.observationID"},
    )
    obscore_default[4] = Column(
        name="obs_collection",
        unit="",
        description="Name of the data collection",
        dtype="U10",
        meta={"Utype": "DataID.collection"},
    )
    obscore_default[5] = Column(
        name="obs_publisher_did",
        unit="",
        description="ID for the Dataset given by the publisher",
        dtype="U30",
        meta={"Utype": "Curation.publisherDID"},
    )
    obscore_default[6] = Column(
        name="access_url",
        unit="",
        description="URL used to access dataset",
        dtype="U30",
        meta={"Utype": "Access.reference"},
    )
    obscore_default[7] = Column(
        name="access_format",
        unit="",
        description="Content format of the dataset",
        dtype="U30",
        meta={"Utype": "Access.format"},
    )
    obscore_default[8] = Column(
        name="access_estsize",
        unit="kbyte",
        description="Estimated size of dataset: in kilobytes",
        dtype="i4",
        meta={"Utype": "Access.size"},
    )
    obscore_default[9] = Column(
        name="s_ra",
        unit="deg",
        description="Central Spatial Position in ICRS Right ascension",
        dtype="f8",
        meta={"Utype": "Char.SpatialAxis.Coverage.Location.Coord.Position2D.Value2.C1"},
    )
    obscore_default[10] = Column(
        name="s_dec",
        unit="deg",
        description="Central Spatial Position in ICRS Declination",
        dtype="f8",
        meta={"Utype": "Char.SpatialAxis.Coverage.Location.Coord.Position2D.Value2.C2"},
    )
    obscore_default[11] = Column(
        name="s_fov",
        unit="deg",
        description="Estimated size of the covered region as the diameter of a containing circle",
        dtype="f8",
        meta={"Utype": "Char.SpatialAxis.Coverage.Bounds.Extent.diameter"},
    )
    obscore_default[12] = Column(
        name="s_region",
        unit="",
        description="Sky region covered by the data product (expressed in ICRS frame)",
        dtype="U30",
        meta={"Utype": "Char.SpatialAxis.Coverage.Support.Area"},
    )
    obscore_default[13] = Column(
        name="s_resolution",
        unit="arcsec",
        description="Spatial resolution of data as FWHM of PSF",
        dtype="f8",
        meta={"Utype": "Char.SpatialAxis.Resolution.refval.value"},
    )
    obscore_default[14] = Column(
        name="s_xel1",
        unit="",
        description="Number of elements along the first coordinate of the spatial axis",
        dtype="i4",
        meta={"Utype": "Char.SpatialAxis.numBins1"},
    )
    obscore_default[15] = Column(
        name="s_xel2",
        unit="",
        description="Number of elements along the second coordinate of the spatial axis",
        dtype="i4",
        meta={"Utype": "Char.SpatialAxis.numBins2"},
    )
    obscore_default[16] = Column(
        name="t_xel",
        unit="",
        description="Number of elements along the time axis",
        dtype="i4",
        meta={"Utype": "Char.TimeAxis.numBins"},
    )
    obscore_default[17] = Column(
        name="t_min",
        unit="d",
        description="Start time in MJD",
        dtype="f8",
        meta={"Utype": "Char.TimeAxis.Coverage.Bounds.Limits.StartTime"},
    )
    obscore_default[18] = Column(
        name="t_max",
        unit="d",
        description="Stop time in MJD",
        dtype="f8",
        meta={"Utype": "Char.TimeAxis.Coverage.Bounds.Limits.StopTime"},
    )
    obscore_default[19] = Column(
        name="t_exptime",
        unit="s",
        description="Total exposure time",
        dtype="f8",
        meta={"Utype": "Char.TimeAxis.Coverage.Support.Extent"},
    )
    obscore_default[20] = Column(
        name="t_resolution",
        unit="s",
        description="Temporal resolution FWHM",
        dtype="f8",
        meta={"Utype": "Char.TimeAxis.Resolution.Refval.valueResolution.Refval.value"},
    )
    obscore_default[21] = Column(
        name="em_xel",
        unit="",
        description="Number of elements along the spectral axis",
        dtype="i4",
        meta={"Utype": "Char.SpectralAxis. numBins"},
    )
    obscore_default[22] = Column(
        name="em_min",
        unit="m",
        description="start in spectral coordinates",
        dtype="f8",
        meta={"Utype": "Char.SpectralAxis.Coverage.Bounds.Limits.LoLimit"},
    )
    obscore_default[23] = Column(
        name="em_max",
        unit="m",
        description="stop in spectral coordinates",
        dtype="f8",
        meta={"Utype": "Char.SpectralAxis.Coverage.Bounds.Limits.HiLimit"},
    )
    obscore_default[24] = Column(
        name="em_res_power",
        unit="",
        description="Value of the resolving power along the spectral axis(R)",
        dtype="f8",
        meta={"Utype": "Char.SpectralAxis.Resolution.ResolPower.refVal"},
    )
    obscore_default[25] = Column(
        name="o_ucd",
        unit="",
        description="Nature of the observable axis",
        dtype="U30",
        meta={"Utype": "Char.ObservableAxis.ucd"},
    )
    obscore_default[26] = Column(
        name="pol_xel",
        unit="",
        description="Number of elements along the polarization axis",
        dtype="i4",
        meta={"Utype": "Char.PolarizationAxis.numBins"},
    )
    obscore_default[27] = Column(
        name="facility_name",
        unit="",
        description="The name of the facility, telescope space craft used for the observation",
        dtype="U10",
        meta={"Utype": "Provenance.ObsConfig.Facility.name"},
    )
    obscore_default[28] = Column(
        name="instrument_name",
        unit="",
        description="The name of the instrument used for the observation",
        dtype="U25",
        meta={"Utype": "Provenance.ObsConfig.Instrument.name"},
    )
    tab_default = Table()
    for var in obscore_default:
        tab_default.add_column(var)
    return tab_default


def _obscore_row(base_dir, single_obsID, obs_publisher_did, access_url, table):
    """Generates an obscore row corresponding to a single obsID

    Parameters
    ----------
    single_obsID : int
        single Observation ID
    **kwargs : `str` {obs_publisher_did, access_url}
    Giving the values for is highly recommended.
    If any of these are not given the corresponding obscore field is left empty and a warning is raised for each empty value.

    Returns
    -------
    table : `~astropy.table.Table`
    """
    base_dir = make_path(base_dir)
    data_store = DataStore.from_dir(base_dir)

    if obs_publisher_did is None:

        log.warning(
            "Insufficient publisher information: 'obs_publisher_did' obscore value will be empty."
        )
    if access_url is None:
        log.warning(
            "Insufficient publisher information: access_url' obscore value will be empty."
        )

    tab = table
    observation = data_store.obs(single_obsID)
    obs_mask = data_store.obs_table["OBS_ID"] == observation.obs_id
    obs_pos = data_store.obs_table[obs_mask]["EVENTS_FILENAME"]
    path = make_path(os.environ["GAMMAPY_DATA"] + "/" + obs_pos[0])
    size = int(os.path.getsize(path) / 1000.0)
    tab.add_row(
        {
            "dataproduct_type": observation.obs_info["EXTNAME"],
            "calib_level": 2,  # Look into the data
            "target_name": observation.obs_info["OBJECT"],
            "obs_id": str(observation.obs_info["OBS_ID"]),
            "obs_collection": "DL3",
            "obs_publisher_did": str(obs_publisher_did),
            "access_url": str(access_url),
            "access_format": "application/fits",
            "access_estsize": size,
            "s_ra": observation.get_pointing_icrs(observation.tmid).ra.to_value("deg"),
            "s_dec": observation.get_pointing_icrs(observation.tmid).dec.to_value(
                "deg"
            ),
            "s_fov": 10.0,
            "t_min": observation.tstart.to_value("mjd"),
            "t_max": observation.tstop.to_value("mjd"),
            "t_exptime": observation.observation_live_time_duration.to_value("s"),
            "em_min": observation.events.energy.min().value,
            "em_max": observation.events.energy.max().value,
            "facility_name": observation.obs_info["TELESCOP"],
            "instrument_name": observation.obs_info["TELLIST"],
        }
    )

    return tab


def to_obscore_table(
    base_dir, selected_obs=None, obs_publisher_did=None, access_url=None
):
    """Generate the complete obscore Table by adding one row per observation using _obscore_row()

    Parameters
    ----------
    selected_obs : list or array of Observation ID(int)
    (default of ``None`` means ``no observation ``)
    If not given, the obscore default table is returned.



    **kwargs : `str` {obs_publisher_did, access_url}
    Giving the values for is highly recommended.
    If any of these are not given the corresponding obscore field is left empty and a warning is raised for each empty value.

    Returns
    -------
    obscore_tab : ~astropy.table.Table
          Obscore table with number of rows = len(selected_obs)
    """

    obscore_tab = _obscore_def()
    for i in range(0, len(selected_obs)):
        obscore_row = _obscore_row(
            base_dir, selected_obs[i], obs_publisher_did, access_url, obscore_tab
        )
        obscore_tab = obscore_row
    return obscore_tab
