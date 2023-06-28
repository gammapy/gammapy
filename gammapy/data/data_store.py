# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import subprocess
from copy import copy
from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import Checker
from .hdu_index_table import HDUIndexTable
from .obs_table import ObservationTable, ObservationTableChecker
from .observations import Observation, ObservationChecker, Observations

__all__ = ["DataStore"]

ALL_IRFS = ["aeff", "edisp", "psf", "bkg", "rad_max"]
ALL_HDUS = ["events", "gti", "pointing"] + ALL_IRFS
REQUIRED_IRFS = {
    "full-enclosure": {"aeff", "edisp", "psf", "bkg"},
    "point-like": {"aeff", "edisp"},
    "all-optional": {},
}


class MissingRequiredHDU(IOError):
    pass


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class DataStore:
    """IACT data store.

    The data selection and access happens using an observation
    and an HDU index file as described at :ref:`gadf:iact-storage`.

    Parameters
    ----------
    hdu_table : `~gammapy.data.HDUIndexTable`
        HDU index table
    obs_table : `~gammapy.data.ObservationTable`
        Observation index table

    Examples
    --------
    Here's an example how to create a `DataStore` to access H.E.S.S. data:

    >>> from gammapy.data import DataStore
    >>> data_store = DataStore.from_dir('$GAMMAPY_DATA/hess-dl3-dr1')
    >>> data_store.info() #doctest: +SKIP
    Data store:
    HDU index table:
    BASE_DIR: /Users/ASinha/Gammapy-dev/gammapy-data/hess-dl3-dr1
    Rows: 630
    OBS_ID: 20136 -- 47829
    HDU_TYPE: ['aeff', 'bkg', 'edisp', 'events', 'gti', 'psf']
    HDU_CLASS: ['aeff_2d', 'bkg_3d', 'edisp_2d', 'events', 'gti', 'psf_table']
    <BLANKLINE>
    <BLANKLINE>
    Observation table:
    Observatory name: 'N/A'
    Number of observations: 105
    <BLANKLINE>

    For further usage example see :doc:`/tutorials/data/cta` tutorial.
    """

    DEFAULT_HDU_TABLE = "hdu-index.fits.gz"
    """Default HDU table filename."""

    DEFAULT_OBS_TABLE = "obs-index.fits.gz"
    """Default observation table filename."""

    def __init__(self, hdu_table=None, obs_table=None):
        self.hdu_table = hdu_table
        self.obs_table = obs_table

    def __str__(self):
        return self.info(show=False)

    @property
    def obs_ids(self):
        """Return the sorted obs_ids contained in the datastore."""
        return np.unique(self.hdu_table["OBS_ID"].data)

    @classmethod
    def from_file(cls, filename, hdu_hdu="HDU_INDEX", hdu_obs="OBS_INDEX"):
        """Create a Datastore from a FITS file.

        The FITS file must contain both index files.

        Parameters
        ----------
        filename : str, Path
            FITS filename
        hdu_hdu : str or int
            FITS HDU name or number for the HDU index table
        hdu_obs : str or int
            FITS HDU name or number for the observation index table

        Returns
        -------
        data_store : `DataStore`
            Data store
        """
        filename = make_path(filename)

        hdu_table = HDUIndexTable.read(filename, hdu=hdu_hdu, format="fits")

        obs_table = None
        if hdu_obs:
            obs_table = ObservationTable.read(filename, hdu=hdu_obs, format="fits")

        return cls(hdu_table=hdu_table, obs_table=obs_table)

    @classmethod
    def from_dir(cls, base_dir, hdu_table_filename=None, obs_table_filename=None):
        """Create from a directory.

        Parameters
        ----------
        base_dir : str, Path
            Base directory of the data files.
        hdu_table_filename : str, Path
            Filename of the HDU index file. May be specified either relative
            to `base_dir` or as an absolute path. If None, the default filename
            will be looked for.
        obs_table_filename : str, Path
            Filename of the observation index file. May be specified either relative
            to `base_dir` or as an absolute path. If None, the default filename
            will be looked for.

        Returns
        -------
        data_store : `DataStore`
            Data store

        Examples
        --------
        >>> from gammapy.data import DataStore
        >>> data_store = DataStore.from_dir('$GAMMAPY_DATA/hess-dl3-dr1')
        """

        base_dir = make_path(base_dir)

        if hdu_table_filename:
            hdu_table_filename = make_path(hdu_table_filename)
            if (base_dir / hdu_table_filename).exists():
                hdu_table_filename = base_dir / hdu_table_filename
        else:
            hdu_table_filename = base_dir / cls.DEFAULT_HDU_TABLE

        if obs_table_filename:
            obs_table_filename = make_path(obs_table_filename)
            if (base_dir / obs_table_filename).exists():
                obs_table_filename = base_dir / obs_table_filename
            elif not obs_table_filename.exists():
                raise IOError(f"File not found : {obs_table_filename}")
        else:
            obs_table_filename = base_dir / cls.DEFAULT_OBS_TABLE

        if not hdu_table_filename.exists():
            raise OSError(f"File not found: {hdu_table_filename}")
        log.debug(f"Reading {hdu_table_filename}")
        hdu_table = HDUIndexTable.read(hdu_table_filename, format="fits")
        hdu_table.meta["BASE_DIR"] = str(base_dir)

        if not obs_table_filename.exists():
            log.info("Cannot find default obs-index table.")
            obs_table = None
        else:
            log.debug(f"Reading {obs_table_filename}")
            obs_table = ObservationTable.read(obs_table_filename, format="fits")

        return cls(hdu_table=hdu_table, obs_table=obs_table)

    @classmethod
    def from_events_files(cls, events_paths, irfs_paths=None):
        """Create from a list of event filenames.

        HDU and observation index tables will be created from the EVENTS header.

        IRFs are found only if you have a ``CALDB`` environment variable set,
        and if the EVENTS files contain the following keys:

        - ``TELESCOP`` (example: ``TELESCOP = CTA``)
        - ``CALDB`` (example: ``CALDB = 1dc``)
        - ``IRF`` (example: ``IRF = South_z20_50h``)

        This method is useful specifically if you want to load data simulated
        with `ctobssim`_

        .. _ctobssim: http://cta.irap.omp.eu/ctools/users/reference_manual/ctobssim.html

        Parameters
        ----------
        events_paths : list of str or Path
            List of paths to the events files
        irfs_paths : str, Path, or list of str or Path
            Path to the IRFs file. If a list is provided it must be the same length
            than `events_paths`. If None the events files have to contain CALDB and
            IRF header keywords to locate the IRF files, otherwise the IRFs are
            assumed to be contained in the events files.

        Returns
        -------
        data_store : `DataStore`
            Data store

        Examples
        --------
        This is how you can access a single event list::

        >>> from gammapy.data import DataStore
        >>> import os
        >>> os.environ["CALDB"] = os.environ["GAMMAPY_DATA"] + "/cta-1dc/caldb"
        >>> path = "$GAMMAPY_DATA/cta-1dc/data/baseline/gps/gps_baseline_110380.fits"
        >>> data_store = DataStore.from_events_files([path])
        >>> observations = data_store.get_observations()

        You can now analyse this data as usual (see any Gammapy tutorial).

        If you have multiple event files, you have to make the list. Here's an example
        using ``Path.glob`` to get a list of all events files in a given folder::

        >>> import os
        >>> from pathlib import Path
        >>> path = Path(os.environ["GAMMAPY_DATA"]) / "cta-1dc/data"
        >>> paths = list(path.rglob("*.fits"))
        >>> data_store = DataStore.from_events_files(paths)
        >>> observations = data_store.get_observations()

        >>> #Note that you have a lot of flexibility to select the observations you want,
        >>> # by having a few lines of custom code to prepare ``paths``, or to select a
        >>> # subset via a method on the ``data_store`` or the ``observations`` objects.
        >>> # If you want to generate HDU and observation index files, write the tables to disk::

        >>> data_store.hdu_table.write("hdu-index.fits.gz") # doctest: +SKIP
        >>> data_store.obs_table.write("obs-index.fits.gz") # doctest: +SKIP
        """
        return DataStoreMaker(events_paths, irfs_paths).run()

    def info(self, show=True):
        """Print some info."""
        s = "Data store:\n"
        s += self.hdu_table.summary()
        s += "\n\n"
        if self.obs_table:
            s += self.obs_table.summary()
        else:
            s += "No observation index table."

        if show:
            print(s)
        else:
            return s

    def obs(self, obs_id, required_irf="full-enclosure", require_events=True):
        """Access a given `~gammapy.data.Observation`.

        Parameters
        ----------
        obs_id : int
            Observation ID.
        required_irf : list of str or str
            The list can include the following options:

            * `"events"` : Events
            * `"gti"` :  Good time intervals
            * `"aeff"` : Effective area
            * `"bkg"` : Background
            * `"edisp"`: Energy dispersion
            * `"psf"` : Point Spread Function
            * `"rad_max"` : Maximal radius

            Alternatively single string can be used as shortcut:

            * `"full-enclosure"` : includes `["events", "gti", "aeff", "edisp", "psf", "bkg"]`
            * `"point-like"` : includes `["events", "gti", "aeff", "edisp"]`
        require_events : bool
            Require events and gti table or not.

        Returns
        -------
        observation : `~gammapy.data.Observation`
            Observation container

        """
        if obs_id not in self.hdu_table["OBS_ID"]:
            raise ValueError(f"OBS_ID = {obs_id} not in HDU index table.")

        kwargs = {"obs_id": int(obs_id)}

        # check for the "short forms"
        if isinstance(required_irf, str):
            required_irf = REQUIRED_IRFS[required_irf]

        if not set(required_irf).issubset(ALL_IRFS):
            difference = set(required_irf).difference(ALL_IRFS)
            raise ValueError(
                f"{difference} is not a valid hdu key. Choose from: {ALL_IRFS}"
            )

        if require_events:
            required_hdus = {"events", "gti"}.union(required_irf)
        else:
            required_hdus = required_irf

        missing_hdus = []
        for hdu in ALL_HDUS:
            hdu_location = self.hdu_table.hdu_location(
                obs_id=obs_id,
                hdu_type=hdu,
                warn_missing=False,
            )
            if hdu_location is not None:
                kwargs[hdu] = hdu_location
            elif hdu in required_hdus:
                missing_hdus.append(hdu)

        if len(missing_hdus) > 0:
            raise MissingRequiredHDU(
                f"Required HDUs {missing_hdus} not found in observation {obs_id}"
            )

        # TODO: right now, gammapy doesn't support using the pointing table of GADF
        # so we always pass the events location here to be read into a FixedPointingInfo
        if "events" in kwargs:
            pointing_location = copy(kwargs["events"])
            pointing_location.hdu_class = "pointing"
            kwargs["pointing"] = pointing_location

        return Observation(**kwargs)

    def get_observations(
        self,
        obs_id=None,
        skip_missing=False,
        required_irf="full-enclosure",
        require_events=True,
    ):
        """Generate a `~gammapy.data.Observations`.

        Parameters
        ----------
        obs_id : list
            Observation IDs (default of ``None`` means "all")
            If not given, all observations ordered by OBS_ID are returned.
            This is not necessarily the order in the ``obs_table``.
        skip_missing : bool, optional
            Skip missing observations, default: False
        required_irf : list of str or str
            Runs will be added to the list of observations only if the
            required HDUs are present. Otherwise, the given run will be skipped
            The list can include the following options:

            * `"events"` : Events
            * `"gti"` :  Good time intervals
            * `"aeff"` : Effective area
            * `"bkg"` : Background
            * `"edisp"`: Energy dispersion
            * `"psf"` : Point Spread Function
            * `"rad_max"` : Maximal radius

            Alternatively single string can be used as shortcut:

            * `"full-enclosure"` : includes `["events", "gti", "aeff", "edisp", "psf", "bkg"]`
            * `"point-like"` : includes `["events", "gti", "aeff", "edisp"]`
            * `"all-optional"` : no HDUs are required, only warnings will be emitted
              for missing HDUs among all possibilities.

        require_events : bool
            Require events and gti table or not.

        Returns
        -------
        observations : `~gammapy.data.Observations`
            Container holding a list of `~gammapy.data.Observation`
        """

        if obs_id is None:
            obs_id = self.obs_ids

        obs_list = []

        for _ in progress_bar(obs_id, desc="Obs Id"):
            try:
                obs = self.obs(_, required_irf, require_events)
            except ValueError as err:
                if skip_missing:
                    log.warning(f"Skipping missing obs_id: {_!r}")
                    continue
                else:
                    raise err
            except MissingRequiredHDU as e:
                log.warning(f"Skipping run with missing HDUs; {e}")
                continue

            obs_list.append(obs)

        log.info(f"Observations selected: {len(obs_list)} out of {len(obs_id)}.")
        return Observations(obs_list)

    def copy_obs(self, obs_id, outdir, hdu_class=None, verbose=False, overwrite=False):
        """Create a new `~gammapy.data.DataStore` containing a subset of observations.

        Parameters
        ----------
        obs_id : array-like, `~gammapy.data.ObservationTable`
            List of observations to copy
        outdir : str, Path
            Directory for the new store
        hdu_class : list of str
            see :attr:`gammapy.data.HDUIndexTable.VALID_HDU_CLASS`
        verbose : bool
            Print copied files
        overwrite : bool
            Overwrite
        """
        outdir = make_path(outdir)

        if not outdir.is_dir():
            raise OSError(f"Not a directory: outdir={outdir}")

        if isinstance(obs_id, ObservationTable):
            obs_id = obs_id["OBS_ID"].data

        hdutable = self.hdu_table
        hdutable.add_index("OBS_ID")
        with hdutable.index_mode("discard_on_copy"):
            subhdutable = hdutable.loc[obs_id]
        if hdu_class is not None:
            subhdutable.add_index("HDU_CLASS")
            with subhdutable.index_mode("discard_on_copy"):
                subhdutable = subhdutable.loc[hdu_class]
        if self.obs_table:
            subobstable = self.obs_table.select_obs_id(obs_id)

        for idx in range(len(subhdutable)):
            # Changes to the file structure could be made here
            loc = subhdutable.location_info(idx)
            targetdir = outdir / loc.file_dir
            targetdir.mkdir(exist_ok=True, parents=True)
            cmd = ["cp"]
            if verbose:
                cmd += ["-v"]
            if not overwrite:
                cmd += ["-n"]
            cmd += [str(loc.path()), str(targetdir)]
            subprocess.run(cmd)

        filename = outdir / self.DEFAULT_HDU_TABLE
        subhdutable.write(filename, format="fits", overwrite=overwrite)

        if self.obs_table:
            filename = outdir / self.DEFAULT_OBS_TABLE
            subobstable.write(str(filename), format="fits", overwrite=overwrite)

    def check(self, checks="all"):
        """Check index tables and data files.

        This is a generator that yields a list of dicts.
        """
        checker = DataStoreChecker(self)
        return checker.run(checks=checks)

<<<<<<< HEAD

    def _obscore_def(self):
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
        )
        obscore_default[1] = Column(
            name="calib_level",
            unit="",
            description="Calibration level of the observation: in {0, 1, 2, 3, 4}",
            dtype="i4",
        )
        obscore_default[2] = Column(
            name="target_name", unit="", description="Object of interest", dtype="U25"
        )
        obscore_default[3] = Column(
            name="obs_id",
            unit="",
            description="Internal  ID given by the ObsTAP service",
            dtype="U10",
        )
        obscore_default[4] = Column(
            name="obs_collection",
            unit="",
            description="Name of the data collection",
            dtype="U10",
        )
        obscore_default[5] = Column(
            name="obs_publisher_did",
            unit="",
            description="ID for the Dataset   given by the publisher",
            dtype="U30",
        )
        obscore_default[6] = Column(
            name="access_url",
            unit="",
            description="URL used to access dataset",
            dtype="U30",
        )
        obscore_default[7] = Column(
            name="access_format",
            unit="",
            description="Content format of the dataset",
            dtype="U30",
        )
        obscore_default[8] = Column(
            name="access_estsize",
            unit="kbyte",
            description="Estimated size of dataset: in kilobytes",
            dtype="i4",
        )
        obscore_default[9] = Column(
            name="s_ra",
            unit="deg",
            description="Central Spatial Position in ICRS Right ascension",
            dtype="f8",
        )
        obscore_default[10] = Column(
            name="s_dec",
            unit="deg",
            description="Central Spatial Position in ICRS Declination",
            dtype="f8",
        )
        obscore_default[11] = Column(
            name="s_fov",
            unit="deg",
            description="Estimated size of the covered region as the diameter of a containing circle",
            dtype="f8",
        )
        obscore_default[12] = Column(
            name="s_region",
            unit="",
            description="Sky region covered by the  data product (expressed in ICRS frame)",
            dtype="U30",
        )
        obscore_default[13] = Column(
            name="s_resolution",
            unit="arcsec",
            description="Spatial resolution of data as FWHM of PSF",
            dtype="f8",
        )
        obscore_default[14] = Column(
            name="s_xel1",
            unit="",
            description="Number of elements along the first coordinate of the spatial  axis",
            dtype="i4",
        )  # ? problem with null and int (for now leave it as str)
        obscore_default[15] = Column(
            name="s_xel2",
            unit="",
            description="Number of elements along the second coordinate of the spatial  axis",
            dtype="i4",
        )  # ? problem with null and int (for now leave it as str)
        obscore_default[16] = Column(
            name="t_xel",
            unit="",
            description="Number of elements along the time axis",
            dtype="i4",
        )  # ? problem with null and int (for now leave it as str)
        obscore_default[17] = Column(
            name="t_min", unit="d", description="Start time in MJD", dtype="f8"
        )
        obscore_default[18] = Column(
            name="t_max", unit="d", description="Stop time  in MJD", dtype="f8"
        )
        obscore_default[19] = Column(
            name="t_exptime", unit="s", description="Total exposure time", dtype="f8"
        )
        obscore_default[20] = Column(
            name="t_resolution",
            unit="s",
            description="Temporal resolution FWHM",
            dtype="f8",
        )
        obscore_default[21] = Column(
            name="em_xel",
            unit="",
            description="Number of elements along the spectral axis",
            dtype="i4",
        )
        obscore_default[22] = Column(
            name="em_min",
            unit="m",
            description="start in spectral coordinates",
            dtype="f8",
        )
        obscore_default[23] = Column(
            name="em_max",
            unit="m",
            description="stop in spectral coordinates",
            dtype="f8",
        )
        obscore_default[24] = Column(
            name="em_res_power",
            unit="",
            description="Value of the resolving power along the spectral axis (R)",
            dtype="f8",
        )
        obscore_default[25] = Column(
            name="o_ucd",
            unit="",
            description="Nature of the observable axis",
            dtype="U30",
        )
        obscore_default[26] = Column(
            name="pol_xel",
            unit="",
            description="Number of elements along the polarization axis",
            dtype="i4",
        )  # ? problem with null and int (for now leave it as str)
        obscore_default[27] = Column(
            name="facility_name",
            unit="",
            description="The name of the facility, telescope space craft used for the observation",
            dtype="U10",
        )
        obscore_default[28] = Column(
            name="instrument_name",
            unit="",
            description="The name of the instrument used for the observation",
            dtype="U25",
        )

        tab_default = Table()
        for var in obscore_default:
            tab_default.add_column(var)
        return tab_default

    def _obscore_row(self, single_obsID, **kwargs):
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

        tab = self._obscore_def()
        tab.add_row()

        observation = self.obs(single_obsID)

        obs_mask = self.obs_table["OBS_ID"] == observation.obs_id
        obs_pos = self.obs_table[obs_mask]["EVENTS_FILENAME"]
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
        #    tab["s_region"] = 'null'
        #    tab["s_resolution"] = np.nan
        #    tab["s_xel1"] = 'null'
        #    tab["s_xel2"] = 'null'
        #    tab["t_xel"] = 'null'
        tab["t_min"] = observation.tstart.value
        tab["t_max"] = observation.tstop.value
        tab["t_exptime"] = observation.observation_live_time_duration.value
        #    tab["t_resolution"] = np.nan
        #    tab["em_xel"] = 'null'
        tab["em_min"] = observation.events.energy.min().value
        tab["em_max"] = observation.events.energy.max().value
        #    tab["em_res_power"] = np.nan
        #    tab["o_ucd"] = 'null'
        #    tab["pol_xel"] = 'null'
        tab["facility_name"] = observation.obs_info["TELESCOP"]
        tab["instrument_name"] = observation.obs_info["TELLIST"]

        return tab

    def to_obscore_table(self, selected_obs=None, format=None, **kwargs):

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

        tab_default = self._obscore_def()

        if selected_obs is None:
            obscore_tab = tab_default

        else:
            obscore_tab = Table()
            for i in range(0, len(selected_obs)):
                obscore_row = self._obscore_row(selected_obs[i], **kwargs)
                obscore_tab = vstack([obscore_tab, obscore_row])

        self._write_obscore_file(obscore_tab, format)

        return obscore_tab

    def _write_obscore_file(self, obscore_tab, format):

        """Write Obscore Table length = len(selected_obs) named 'obscore_table'

        Parameters
        ----------
            obscore_tab : Astropy Table
            format : Astropy Table format
                Define the format for the output obscore table
                (default of ``None`` means ``.fits``)

        """

        if format:

            obscore_tab.write("obscore_table", format=format, overwrite=True)
        else:
            obscore_tab.write("obscore_table.fit", format="fits", overwrite=True)
        return



=======
>>>>>>> d32f8ffac (I moved the functions that were in datastore to a new script in gammapy/data/ivoa.py. Now they all work as independent functions but the only one accessible to the user is to_obscore_table. This function returns an astropy obscore table)

class DataStoreChecker(Checker):
    """Check data store.

    Checks data format and a bit about the content.
    """

    CHECKS = {
        "obs_table": "check_obs_table",
        "hdu_table": "check_hdu_table",
        "observations": "check_observations",
        "consistency": "check_consistency",
    }

    def __init__(self, data_store):
        self.data_store = data_store

    def check_obs_table(self):
        """Checks for the observation index table."""
        yield from ObservationTableChecker(self.data_store.obs_table).run()

    def check_hdu_table(self):
        """Checks for the HDU index table."""
        t = self.data_store.hdu_table
        m = t.meta
        if m.get("HDUCLAS1", "") != "INDEX":
            yield {
                "level": "error",
                "hdu": "hdu-index",
                "msg": "Invalid header key. Must have HDUCLAS1=INDEX",
            }
        if m.get("HDUCLAS2", "") != "HDU":
            yield {
                "level": "error",
                "hdu": "hdu-index",
                "msg": "Invalid header key. Must have HDUCLAS2=HDU",
            }

        # Check that all HDU in the data files exist
        for idx in range(len(t)):
            location_info = t.location_info(idx)
            try:
                location_info.get_hdu()
            except KeyError:
                yield {
                    "level": "error",
                    "msg": f"HDU not found: {location_info.__dict__!r}",
                }

    def check_consistency(self):
        """Check consistency between multiple HDUs."""
        # obs and HDU index should have the same OBS_ID
        obs_table_obs_id = set(self.data_store.obs_table["OBS_ID"])
        hdu_table_obs_id = set(self.data_store.hdu_table["OBS_ID"])
        if not obs_table_obs_id == hdu_table_obs_id:
            yield {
                "level": "error",
                "msg": "Inconsistent OBS_ID in obs and HDU index tables",
            }
        # TODO: obs table and events header should have the same times

    def check_observations(self):
        """Perform some sanity checks for all observations."""
        for obs_id in self.data_store.obs_table["OBS_ID"]:
            obs = self.data_store.obs(obs_id)
            yield from ObservationChecker(obs).run()


class DataStoreMaker:
    """Create data store index tables.

    This is a multi-step process coded as a class.
    Users will usually call this via `DataStore.from_events_files`.
    """

    def __init__(self, events_paths, irfs_paths=None):
        if isinstance(events_paths, (str, Path)):
            raise TypeError("Need list of paths, not a single string or Path object.")

        self.events_paths = [make_path(path) for path in events_paths]
        if irfs_paths is None or isinstance(irfs_paths, (str, Path)):
            self.irfs_paths = [make_path(irfs_paths)] * len(events_paths)
        else:
            self.irfs_paths = [make_path(path) for path in irfs_paths]

        # Cache for EVENTS file header information, to avoid multiple reads
        self._events_info = {}

    def run(self):
        hdu_table = self.make_hdu_table()
        obs_table = self.make_obs_table()
        return DataStore(hdu_table=hdu_table, obs_table=obs_table)

    def get_events_info(self, events_path, irf_path=None):
        if events_path not in self._events_info:
            self._events_info[events_path] = self.read_events_info(
                events_path, irf_path
            )

        return self._events_info[events_path]

    def get_obs_info(self, events_path, irf_path=None):
        # We could add or remove info here depending on what we want in the obs table
        return self.get_events_info(events_path, irf_path)

    @staticmethod
    def read_events_info(events_path, irf_path=None):
        """Read mandatory events header info"""
        log.debug(f"Reading {events_path}")

        with fits.open(events_path, memmap=False) as hdu_list:
            header = hdu_list["EVENTS"].header

        na_int, na_str = -1, "NOT AVAILABLE"

        info = {}
        # Note: for some reason `header["OBS_ID"]` is sometimes `str`, maybe trailing whitespace
        # mandatory header info:
        info["OBS_ID"] = int(header["OBS_ID"])
        info["TSTART"] = header["TSTART"] * u.s
        info["TSTOP"] = header["TSTOP"] * u.s
        info["ONTIME"] = header["ONTIME"] * u.s
        info["LIVETIME"] = header["LIVETIME"] * u.s
        info["DEADC"] = header["DEADC"]
        info["TELESCOP"] = header.get("TELESCOP", na_str)

        obs_mode = header.get("OBS_MODE", "POINTING")
        if obs_mode == "DRIFT":
            info["ALT_PNT"] = header["ALT_PNT"] * u.deg
            info["AZ_PNT"] = header["AZ_PNT"] * u.deg
            info["ZEN_PNT"] = 90 * u.deg - info["ALT_PNT"]
        else:
            info["RA_PNT"] = header["RA_PNT"] * u.deg
            info["DEC_PNT"] = header["DEC_PNT"] * u.deg

        # optional header info
        pos = SkyCoord(info["RA_PNT"], info["DEC_PNT"], unit="deg").galactic
        info["GLON_PNT"] = pos.l
        info["GLAT_PNT"] = pos.b
        info["DATE-OBS"] = header.get("DATE_OBS", na_str)
        info["TIME-OBS"] = header.get("TIME_OBS", na_str)
        info["DATE-END"] = header.get("DATE_END", na_str)
        info["TIME-END"] = header.get("TIME_END", na_str)
        info["N_TELS"] = header.get("N_TELS", na_int)
        info["OBJECT"] = header.get("OBJECT", na_str)

        # Not part of the spec, but good to know from which file the info comes
        info["EVENTS_FILENAME"] = str(events_path)
        info["EVENT_COUNT"] = header["NAXIS2"]

        # This is the info needed to link from EVENTS to IRFs
        info["CALDB"] = header.get("CALDB", na_str)
        info["IRF"] = header.get("IRF", na_str)
        if irf_path is not None:
            info["IRF_FILENAME"] = str(irf_path)
        elif info["CALDB"] != na_str and info["IRF"] != na_str:
            caldb_irf = CalDBIRF.from_meta(info)
            info["IRF_FILENAME"] = str(caldb_irf.file_path)
        else:
            info["IRF_FILENAME"] = info["EVENTS_FILENAME"]

        # Mandatory fields defining the time data
        for name in tu.TIME_KEYWORDS:
            info[name] = header.get(name, None)

        return info

    def make_obs_table(self):
        rows = []
        time_rows = []
        for events_path, irf_path in zip(self.events_paths, self.irfs_paths):
            row = self.get_obs_info(events_path, irf_path)
            rows.append(row)
            time_row = tu.extract_time_info(row)
            time_rows.append(time_row)

        names = list(rows[0].keys())
        table = ObservationTable(rows=rows, names=names)

        m = table.meta
        if not tu.unique_time_info(time_rows):
            raise RuntimeError(
                "The time information in the EVENT header are not consistent between observations"
            )
        for name in tu.TIME_KEYWORDS:
            m[name] = time_rows[0][name]

        m["HDUCLASS"] = "GADF"
        m["HDUDOC"] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
        m["HDUVERS"] = "0.2"
        m["HDUCLAS1"] = "INDEX"
        m["HDUCLAS2"] = "OBS"

        return table

    def make_hdu_table(self):
        rows = []
        for events_path, irf_path in zip(self.events_paths, self.irfs_paths):
            rows.extend(self.get_hdu_table_rows(events_path, irf_path))

        names = list(rows[0].keys())
        # names = ['OBS_ID', 'HDU_TYPE', 'HDU_CLASS', 'FILE_DIR', 'FILE_NAME', 'HDU_NAME']

        table = HDUIndexTable(rows=rows, names=names)

        m = table.meta
        m["HDUCLASS"] = "GADF"
        m["HDUDOC"] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
        m["HDUVERS"] = "0.2"
        m["HDUCLAS1"] = "INDEX"
        m["HDUCLAS2"] = "HDU"

        return table

    def get_hdu_table_rows(self, events_path, irf_path=None):
        events_info = self.get_obs_info(events_path, irf_path)

        info = dict(
            OBS_ID=events_info["OBS_ID"],
            FILE_DIR=events_path.parent.as_posix(),
            FILE_NAME=events_path.name,
        )
        yield dict(HDU_TYPE="events", HDU_CLASS="events", HDU_NAME="EVENTS", **info)
        yield dict(HDU_TYPE="gti", HDU_CLASS="gti", HDU_NAME="GTI", **info)

        irf_path = Path(events_info["IRF_FILENAME"])
        info = dict(
            OBS_ID=events_info["OBS_ID"],
            FILE_DIR=irf_path.parent.as_posix(),
            FILE_NAME=irf_path.name,
        )
        yield dict(
            HDU_TYPE="aeff", HDU_CLASS="aeff_2d", HDU_NAME="EFFECTIVE AREA", **info
        )
        yield dict(
            HDU_TYPE="edisp", HDU_CLASS="edisp_2d", HDU_NAME="ENERGY DISPERSION", **info
        )
        yield dict(
            HDU_TYPE="psf",
            HDU_CLASS="psf_3gauss",
            HDU_NAME="POINT SPREAD FUNCTION",
            **info,
        )
        yield dict(HDU_TYPE="bkg", HDU_CLASS="bkg_3d", HDU_NAME="BACKGROUND", **info)


# TODO: load IRF file, and infer HDU_CLASS from IRF file contents!
class CalDBIRF:
    """Helper class to work with IRFs in CALDB format."""

    def __init__(self, telescop, caldb, irf):
        self.telescop = telescop
        self.caldb = caldb
        self.irf = irf

    @classmethod
    def from_meta(cls, meta):
        return cls(telescop=meta["TELESCOP"], caldb=meta["CALDB"], irf=meta["IRF"])

    @property
    def file_dir(self):
        # In CTA 1DC the header key is "CTA", but the directory is lower-case "cta"
        telescop = self.telescop.lower()
        return f"$CALDB/data/{telescop}/{self.caldb}/bcf/{self.irf}"

    @property
    def file_path(self):
        return Path(f"{self.file_dir}/{self.file_name}")

    @property
    def file_name(self):
        path = make_path(self.file_dir)
        return list(path.iterdir())[0].name
