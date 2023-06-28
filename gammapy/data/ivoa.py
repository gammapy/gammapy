# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import os
import numpy as np
from astropy.table import Column, Table, vstack
from gammapy.data import DataStore
from gammapy.utils.scripts import make_path

__all__ = ["to_obscore_table"]


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# 	def __init__(self, datastore=None):
# 		self.datastore = datastore


def _obscore_def():
    """Generate the Obscore default table
    In case the obscore standard changes, this function should be changed consistently
    Returns
    -------
    Astropy Table length=0
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
        description="Internal  ID given by the ObsTAP service",
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
        description="ID for the Dataset   given by the publisher",
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
        description="Sky region covered by the  data product (expressed in ICRS frame)",
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
        description="Number of elements along the first coordinate of the spatial  axis",
        dtype="i4",
        meta={"Utype": "Char.SpatialAxis.numBins1"},
    )
    obscore_default[15] = Column(
        name="s_xel2",
        unit="",
        description="Number of elements along the second coordinate of the spatial  axis",
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
        description="Stop time  in MJD",
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
        description="Value of the resolving power along the spectral axis (R)",
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


def _obscore_row(base_dir, single_obsID, **kwargs):
    """Generates an obscore row corresponding to a single obsID
    Parameters
    ----------
    single_obsID : `int`
            single Observation ID
    **kwargs : `str` {obs_publisher_did, access_url}
    Giving the values for is highly recommended.
    If any of these are not given the corresponding obscore field is left empty and a warning is raised for each empty value.

    Returns
    -------
    Astropy Table length=1
    """
    base_dir = make_path(base_dir)
    data_store = DataStore.from_dir(base_dir)
    if kwargs:
        obs_publisher_did = kwargs.get("obs_publisher_did")
        access_url = kwargs.get("access_url")

        if obs_publisher_did is None:

            log.warning(
                "Insufficient publisher information: 'obs_publisher_did' obscore value will be empty."
            )
        if access_url is None:
            log.warning(
                "Insufficient publisher information: access_url' obscore value will be empty."
            )
    else:
        obs_publisher_did = ""
        access_url = ""
        log.warning(
            "Insufficient publisher information: 'obs_publisher_did' and 'access_url' obscore values will be empty."
        )

    tab = _obscore_def()
    tab.add_row()

    observation = data_store.obs(single_obsID)

    obs_mask = data_store.obs_table["OBS_ID"] == observation.obs_id
    obs_pos = data_store.obs_table[obs_mask]["EVENTS_FILENAME"]
    path = make_path(os.environ["GAMMAPY_DATA"] + "/" + obs_pos[0])
    size = int(os.path.getsize(path) / 1000.0)
    tab["dataproduct_type"] = observation.obs_info["EXTNAME"]
    tab["calib_level"] = 2  # Look into the data
    tab["target_name"] = observation.obs_info["OBJECT"]
    tab["obs_id"] = str(observation.obs_info["OBS_ID"])
    tab["obs_collection"] = "DL3"
    tab["obs_publisher_did"] = obs_publisher_did
    tab["access_url"] = access_url
    tab["access_format"] = "application/fits"
    tab["access_estsize"] = size
    tab["s_ra"] = observation.get_pointing_icrs(observation.tmid).ra.value
    tab["s_dec"] = observation.get_pointing_icrs(observation.tmid).dec.value
    tab["s_fov"] = 10.0
    tab["t_min"] = observation.tstart.value
    tab["t_max"] = observation.tstop.value
    tab["t_exptime"] = observation.observation_live_time_duration.value
    tab["em_min"] = observation.events.energy.min().value
    tab["em_max"] = observation.events.energy.max().value
    tab["facility_name"] = observation.obs_info["TELESCOP"]
    tab["instrument_name"] = observation.obs_info["TELLIST"]

    return tab


def to_obscore_table(base_dir, selected_obs=None, **kwargs):

    """Generate the complete obscore Table by stacking length=1 Tables given by DataStore._obscore_row()

    Parameters
    ----------
    selected_obs : list or array of Observation ID (int)
    (default of ``None`` means ``no obaservation ``)
    If not given, the obscore default table is returned.
    format : Astropy Table format
    Define the format for the output obscore table


    **kwargs : `str` {obs_publisher_did, access_url}
    Giving the values for is highly recommended.
    If any of these are not given the corresponding obscore field is left empty and a warning is raised for each empty value.

    Returns
    -------
    Astropy Table length = len(selected_obs)
    """

    tab_default = _obscore_def()

    if selected_obs is None:
        obscore_tab = tab_default

    else:
        obscore_tab = Table()
        for i in range(0, len(selected_obs)):
            obscore_row = _obscore_row(base_dir, selected_obs[i], **kwargs)
            obscore_tab = vstack([obscore_tab, obscore_row])

    return obscore_tab
