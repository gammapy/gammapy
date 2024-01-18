# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import os
import numpy as np
from astropy.table import Column, Table
from gammapy.data import DataStore
from gammapy.utils.scripts import make_path

__all__ = ["to_obscore_table", "obscore_structure"]


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DEFAULT_OBSCORE_TEMPLATE = {
    "dataproduct_type": "event",
    "calib_level": 2,
    "obs_collection": "DL3",
    "access_format": "application/fits",
    "s_fov": 10.0,
}


def obscore_structure():
    """Generate the Obscore default table
    In case the obscore standard changes, this function should be changed according to https://www.ivoa.net/documents/ObsCore

    Returns
    -------
    table : `~astropy.table.Table`
       the empty table
    """
    n_obscore_val = 29  # Number of obscore values
    obscore_table = np.empty(n_obscore_val, dtype=object)
    obscore_table[0] = Column(
        name="dataproduct_type",
        unit="",
        description="Data product (file content) primary type",
        dtype="U10",
        meta={"Utype": "ObsDataset.dataProductType", "UCD": "meta.id"},
    )
    obscore_table[1] = Column(
        name="calib_level",
        unit="",
        description="Calibration level of the observation: in {0, 1, 2, 3, 4}",
        dtype="i4",
        meta={"Utype": "ObsDataset.calibLevel", "UCD": "meta.code;obs.calib"},
    )
    obscore_table[2] = Column(
        name="target_name",
        unit="",
        description="Object of interest",
        dtype="U25",
        meta={"Utype": "Target.name", "UCD": "meta.id;src"},
    )
    obscore_table[3] = Column(
        name="obs_id",
        unit="",
        description="Internal ID given by the ObsTAP service",
        dtype="U10",
        meta={"Utype": "DataID.observationID", "UCD": "meta.id"},
    )
    obscore_table[4] = Column(
        name="obs_collection",
        unit="",
        description="Name of the data collection",
        dtype="U10",
        meta={"Utype": "DataID.collection", "UCD": "meta.id"},
    )
    obscore_table[5] = Column(
        name="obs_publisher_did",
        unit="",
        description="ID for the Dataset given by the publisher",
        dtype="U30",
        meta={"Utype": "Curation.publisherDID", "UCD": "meta.ref.uri;meta.curation"},
    )
    obscore_table[6] = Column(
        name="access_url",
        unit="",
        description="URL used to access dataset",
        dtype="U30",
        meta={"Utype": "Access.reference", "UCD": "meta.ref.url"},
    )
    obscore_table[7] = Column(
        name="access_format",
        unit="",
        description="Content format of the dataset",
        dtype="U30",
        meta={"Utype": "Access.format", "UCD": "meta.code.mime"},
    )
    obscore_table[8] = Column(
        name="access_estsize",
        unit="kbyte",
        description="Estimated size of dataset: in kilobytes",
        dtype="i4",
        meta={"Utype": "Access.size", "UCD": "phys.size;meta.file"},
    )
    obscore_table[9] = Column(
        name="s_ra",
        unit="deg",
        description="Central Spatial Position in ICRS Right ascension",
        dtype="f8",
        meta={
            "Utype": "Char.SpatialAxis.Coverage.Location.Coord.Position2D.Value2.C1",
            "UCD": "pos.eq.ra",
        },
    )
    obscore_table[10] = Column(
        name="s_dec",
        unit="deg",
        description="Central Spatial Position in ICRS Declination",
        dtype="f8",
        meta={
            "Utype": "Char.SpatialAxis.Coverage.Location.Coord.Position2D.Value2.C2",
            "UCD": "pos.eq.dec",
        },
    )
    obscore_table[11] = Column(
        name="s_fov",
        unit="deg",
        description="Estimated size of the covered region as the diameter of a containing circle",
        dtype="f8",
        meta={
            "Utype": "Char.SpatialAxis.Coverage.Bounds.Extent.diameter",
            "UCD": "phys.angSize;instr.fov",
        },
    )
    obscore_table[12] = Column(
        name="s_region",
        unit="",
        description="Sky region covered by the data product (expressed in ICRS frame)",
        dtype="U30",
        meta={
            "Utype": "Char.SpatialAxis.Coverage.Support.Area",
            "UCD": "pos.outline;obs.field",
        },
    )
    obscore_table[13] = Column(
        name="s_resolution",
        unit="arcsec",
        description="Spatial resolution of data as FWHM of PSF",
        dtype="f8",
        meta={
            "Utype": "Char.SpatialAxis.Resolution.refval.value",
            "UCD": "pos.angResolution",
        },
    )
    obscore_table[14] = Column(
        name="s_xel1",
        unit="",
        description="Number of elements along the first coordinate of the spatial axis",
        dtype="i4",
        meta={"Utype": "Char.SpatialAxis.numBins1", "UCD": "meta.number"},
    )
    obscore_table[15] = Column(
        name="s_xel2",
        unit="",
        description="Number of elements along the second coordinate of the spatial axis",
        dtype="i4",
        meta={"Utype": "Char.SpatialAxis.numBins2", "UCD": "meta.number"},
    )
    obscore_table[16] = Column(
        name="t_xel",
        unit="",
        description="Number of elements along the time axis",
        dtype="i4",
        meta={"Utype": "Char.TimeAxis.numBins", "UCD": "meta.number"},
    )
    obscore_table[17] = Column(
        name="t_min",
        unit="d",
        description="Start time in MJD",
        dtype="f8",
        meta={
            "Utype": "Char.TimeAxis.Coverage.Bounds.Limits.StartTime",
            "UCD": "time.start;obs.exposure",
        },
    )
    obscore_table[18] = Column(
        name="t_max",
        unit="d",
        description="Stop time in MJD",
        dtype="f8",
        meta={
            "Utype": "Char.TimeAxis.Coverage.Bounds.Limits.StopTime",
            "UCD": "time.end;obs.exposure",
        },
    )
    obscore_table[19] = Column(
        name="t_exptime",
        unit="s",
        description="Total exposure time",
        dtype="f8",
        meta={
            "Utype": "Char.TimeAxis.Coverage.Support.Extent",
            "UCD": "time.duration;obs.exposure",
        },
    )
    obscore_table[20] = Column(
        name="t_resolution",
        unit="s",
        description="Temporal resolution FWHM",
        dtype="f8",
        meta={
            "Utype": "Char.TimeAxis.Resolution.Refval.valueResolution.Refval.value",
            "UCD": "time.resolution",
        },
    )
    obscore_table[21] = Column(
        name="em_xel",
        unit="",
        description="Number of elements along the spectral axis",
        dtype="i4",
        meta={"Utype": "Char.SpectralAxis. numBins", "UCD": "meta.number"},
    )
    obscore_table[22] = Column(
        name="em_min",
        unit="TeV",
        description="start in spectral coordinates",
        dtype="f8",
        meta={
            "Utype": "Char.SpectralAxis.Coverage.Bounds.Limits.LoLimit",
            "UCD": "em.wl;stat.min",
        },
    )
    obscore_table[23] = Column(
        name="em_max",
        unit="TeV",
        description="stop in spectral coordinates",
        dtype="f8",
        meta={
            "Utype": "Char.SpectralAxis.Coverage.Bounds.Limits.HiLimit",
            "UCD": "em.wl;stat.max",
        },
    )
    obscore_table[24] = Column(
        name="em_res_power",
        unit="",
        description="Value of the resolving power along the spectral axis(R)",
        dtype="f8",
        meta={
            "Utype": "Char.SpectralAxis.Resolution.ResolPower.refVal",
            "UCD": "spect.resolution",
        },
    )
    obscore_table[25] = Column(
        name="o_ucd",
        unit="",
        description="Nature of the observable axis",
        dtype="U30",
        meta={"Utype": "Char.ObservableAxis.ucd", "UCD": "meta.ucd"},
    )
    obscore_table[26] = Column(
        name="pol_xel",
        unit="",
        description="Number of elements along the polarization axis",
        dtype="i4",
        meta={"Utype": "Char.PolarizationAxis.numBins", "UCD": "meta.number"},
    )
    obscore_table[27] = Column(
        name="facility_name",
        unit="",
        description="The name of the facility, telescope space craft used for the observation",
        dtype="U10",
        meta={
            "Utype": "Provenance.ObsConfig.Facility.name",
            "UCD": "meta.id;instr.tel",
        },
    )
    obscore_table[28] = Column(
        name="instrument_name",
        unit="",
        description="The name of the instrument used for the observation",
        dtype="U25",
        meta={"Utype": "Provenance.ObsConfig.Instrument.name", "UCD": "meta.id;instr"},
    )
    tab_default = Table()
    for var in obscore_table:
        tab_default.add_column(var)
    return tab_default


def _obscore_row(
    base_dir,
    single_obsID,
    obs_publisher_did,
    access_url,
    table,
    obscore_template=DEFAULT_OBSCORE_TEMPLATE,
):
    """Generates an obscore row corresponding to a single obsID

    Parameters
    ----------
    base_dir : str or `~pathlib.Path`
        Base directory of the data files.
    single_obsID : int
        single Observation ID
    obs_publisher_did : str, optional
        ID for the Dataset given by the publisher.
        Default is None.
    access_url : str, optional
        URL used to download dataset.
        Default is None.
    obscore_template : dict, optional
        Template for fixed values in the obscore Table.
        Default is DEFAULT_OBSCORE_TEMPLATE
    table : `~astropy.table.Table` with n rows
    Returns
    -------
    tab : `~astropy.table.Table` with n+1 rows
    """
    base_dir = make_path(base_dir)
    data_store = DataStore.from_dir(base_dir)

    tab = table
    observation = data_store.obs(single_obsID)
    obs_mask = (data_store.hdu_table["OBS_ID"] == observation.obs_id) & (
        data_store.hdu_table["HDU_TYPE"] == "events"
    )

    hdu_dir = data_store.hdu_table.base_dir
    file_dir = data_store.hdu_table[obs_mask]["FILE_DIR"]
    file_name = data_store.hdu_table[obs_mask]["FILE_NAME"]
    path = make_path(str(hdu_dir) + "/" + str(file_dir[0]) + "/" + str(file_name[0]))
    size = int(os.path.getsize(path.as_posix()) / 1000.0)
    tab.add_row(
        {
            "dataproduct_type": obscore_template.get(
                "dataproduct_type", DEFAULT_OBSCORE_TEMPLATE["dataproduct_type"]
            ),
            "calib_level": obscore_template.get(
                "calib_level", DEFAULT_OBSCORE_TEMPLATE["calib_level"]
            ),
            "target_name": observation.meta.target.name,
            "obs_id": str(observation.meta.obs_info.obs_id),
            "obs_collection": obscore_template.get(
                "obs_collection", DEFAULT_OBSCORE_TEMPLATE["obs_collection"]
            ),
            "obs_publisher_did": str(obs_publisher_did)
            + "#"
            + str(observation.meta.obs_info.obs_id),
            "access_url": str(access_url) + str(file_name[0]),
            "access_format": obscore_template.get(
                "access_format", DEFAULT_OBSCORE_TEMPLATE["access_format"]
            ),
            "access_estsize": size,
            "s_ra": observation.get_pointing_icrs(observation.tmid).ra.to_value("deg"),
            "s_dec": observation.get_pointing_icrs(observation.tmid).dec.to_value(
                "deg"
            ),
            "s_fov": obscore_template.get("s_fov", DEFAULT_OBSCORE_TEMPLATE["s_fov"]),
            "t_min": observation.tstart.to_value("mjd"),
            "t_max": observation.tstop.to_value("mjd"),
            "t_exptime": observation.observation_live_time_duration.to_value("s"),
            "em_min": observation.events.energy.min().value,
            "em_max": observation.events.energy.max().value,
            "facility_name": observation.meta.obs_info.telescope,
            "instrument_name": observation.aeff.meta["INSTRUME"],
        }
    )

    return tab


def to_obscore_table(
    base_dir,
    selected_obs=None,
    obs_publisher_did=None,
    access_url=None,
    obscore_template=DEFAULT_OBSCORE_TEMPLATE,
):
    """Generate the complete obscore Table by adding one row per observation using _obscore_row()

    Parameters
    ----------
    base_dir : str or `~pathlib.Path`
        Base directory of the data files.
    selected_obs : list or array of Observation ID(int)
        Default is None (default of ``None`` means ``no observation ``).
        If not given, the full obscore (for all the obs_ids in DataStore) table is returned.
    obs_publisher_did : str, optional
        ID for the Dataset given by the publisher.
        Default is None. Giving the values of this argument is highly recommended.
        If not the corresponding obscore field is filled by the Observation ID value and a warning is raised.
    access_url : str, optional
        URL used to download dataset.
        Default is None. Giving the values of this argument is highly recommended.
        If not the corresponding obscore field is filled by the Observation ID value and a warning is raised.
    obscore_template : dict, optional
        Template for fixed values in the obscore Table.
        Default is DEFAULT_OBSCORE_TEMPLATE

    Returns
    -------
    obscore_tab : ~astropy.table.Table
        Obscore table with number of rows = len(selected_obs)
    """

    if obs_publisher_did is None:
        log.warning(
            "Insufficient publisher information: 'obs_publisher_did' obscore value will be empty. Giving this values is highly recommended. obs_publisher_did is the Dataset identifier given by the publisher. For more information about these arguments please check the IVOA recommendations: https://www.ivoa.net/documents/ObsCore and https://www.ivoa.net/documents/IVOAIdentifiers/20160523/REC-Identifiers-2.0.html"
        )
        obs_publisher_did = ""
    if access_url is None:
        log.warning(
            "Insufficient publisher information: access_url' obscore value will be empty. Giving this values is highly recommended. access_url is the URL used to access (download) dataset. For more information about these arguments please check the IVOA recommendations: https://www.ivoa.net/documents/ObsCore and https://www.ivoa.net/documents/TAP/"
        )
        access_url = ""
    if obscore_template is None:
        log.info("No template provided, using DEFAULT_OBSCORE_TEMPLATE")
        obscore_template = DEFAULT_OBSCORE_TEMPLATE

    if not isinstance(obscore_template, dict):
        log.warning("Template is not a dictionary, using DEFAULT_OBSCORE_TEMPLATE ")
        obscore_template = DEFAULT_OBSCORE_TEMPLATE

    if selected_obs is None:
        data_store = DataStore.from_dir(base_dir)
        selected_obs = data_store.obs_ids

    obscore_tab = obscore_structure()
    for i in range(0, len(selected_obs)):
        obscore_row = _obscore_row(
            base_dir,
            selected_obs[i],
            obs_publisher_did,
            access_url,
            obscore_tab,
            obscore_template,
        )
        obscore_tab = obscore_row
    return obscore_tab
