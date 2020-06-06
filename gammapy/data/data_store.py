# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import subprocess
from pathlib import Path
from astropy.coordinates import SkyCoord
from astropy.io import fits
from gammapy.utils.scripts import make_path
from gammapy.utils.table import table_row_to_dict
from gammapy.utils.testing import Checker
from .hdu_index_table import HDUIndexTable
from .obs_table import ObservationTable, ObservationTableChecker
from .observations import Observation, ObservationChecker, Observations

__all__ = ["DataStore"]

log = logging.getLogger(__name__)


class DataStore:
    """IACT data store.

    The data selection and access happens using an observation
    and an HDU index file as described at :ref:`gadf:iact-storage`.

    For a usage example see `cta.html <../tutorials/cta.html>`__

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
    >>> data_store.info()
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

    @classmethod
    def from_file(cls, filename, hdu_hdu="HDU_INDEX", hdu_obs="OBS_INDEX"):
        """Create from a FITS file.

        The FITS file must contain both index files.

        Parameters
        ----------
        filename : str, Path
            FITS filename
        hdu_hdu : str or int
            FITS HDU name or number for the HDU index table
        hdu_obs : str or int
            FITS HDU name or number for the observation index table
        """
        filename = make_path(filename)

        hdu_table = HDUIndexTable.read(filename, hdu=hdu_hdu, format="fits")

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
        else:
            obs_table_filename = base_dir / cls.DEFAULT_OBS_TABLE

        if not hdu_table_filename.exists():
            raise OSError(f"File not found: {hdu_table_filename}")
        log.debug(f"Reading {hdu_table_filename}")
        hdu_table = HDUIndexTable.read(hdu_table_filename, format="fits")
        hdu_table.meta["BASE_DIR"] = str(base_dir)

        if not obs_table_filename.exists():
            raise OSError(f"File not found: {obs_table_filename}")
        log.debug(f"Reading {obs_table_filename}")
        obs_table = ObservationTable.read(obs_table_filename, format="fits")

        return cls(hdu_table=hdu_table, obs_table=obs_table)

    @classmethod
    def from_events_files(cls, paths):
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

        Examples
        --------
        This is how you can access a single event list::

            from gammapy.data import DataStore
            path = "$GAMMAPY_DATA/cta-1dc/data/baseline/gps/gps_baseline_110380.fits"
            data_store = DataStore.from_events_files([path])
            observations = data_store.get_observations()

        You can now analyse this data as usual (see any Gammapy tutorial).

        If you have multiple event files, you have to make the list. Here's an example
        using ``Path.glob`` to get a list of all events files in a given folder::

            import os
            from pathlib import Path
            path = Path(os.environ["GAMMAPY_DATA"]) / "cta-1dc/data"
            paths = list(path.rglob("*.fits"))
            data_store = DataStore.from_events_files(paths)
            observations = data_store.get_observations()

        Note that you have a lot of flexibility to select the observations you want,
        by having a few lines of custom code to prepare ``paths``, or to select a
        subset via a method on the ``data_store`` or the ``observations`` objects.

        If you want to generate HDU and observation index files, write the tables to disk::

            data_store.hdu_table.write("hdu-index.fits.gz")
            data_store.obs_table.write("obs-index.fits.gz")
        """
        return DataStoreMaker(paths).run()

    def info(self, show=True):
        """Print some info."""
        s = "Data store:\n"
        s += self.hdu_table.summary()
        s += "\n\n"
        s += self.obs_table.summary()

        if show:
            print(s)
        else:
            return s

    def obs(self, obs_id):
        """Access a given `~gammapy.data.Observation`.

        Parameters
        ----------
        obs_id : int
            Observation ID.

        Returns
        -------
        observation : `~gammapy.data.Observation`
            Observation container
        """
        if obs_id not in self.obs_table["OBS_ID"]:
            raise ValueError(f"OBS_ID = {obs_id} not in obs index table.")

        if obs_id not in self.hdu_table["OBS_ID"]:
            raise ValueError(f"OBS_ID = {obs_id} not in HDU index table.")

        row = self.obs_table.select_obs_id(obs_id=obs_id)[0]
        obs_info = table_row_to_dict(row)

        aeff_hdu = self.hdu_table.hdu_location(obs_id=obs_id, hdu_type="aeff")
        edisp_hdu = self.hdu_table.hdu_location(obs_id=obs_id, hdu_type="edisp")
        bkg_hdu = self.hdu_table.hdu_location(obs_id=obs_id, hdu_type="bkg")
        psf_hdu = self.hdu_table.hdu_location(obs_id=obs_id, hdu_type="psf")
        events_hdu = self.hdu_table.hdu_location(obs_id=obs_id, hdu_type="events")
        gti_hdu = self.hdu_table.hdu_location(obs_id=obs_id, hdu_type="gti")

        return Observation(
            obs_id=int(obs_id),
            obs_info=obs_info,
            bkg=bkg_hdu,
            aeff=aeff_hdu,
            edisp=edisp_hdu,
            events=events_hdu,
            gti=gti_hdu,
            psf=psf_hdu,
        )

    def get_observations(self, obs_id=None, skip_missing=False):
        """Generate a `~gammapy.data.Observations`.

        Parameters
        ----------
        obs_id : list
            Observation IDs (default of ``None`` means "all")
        skip_missing : bool, optional
            Skip missing observations, default: False

        Returns
        -------
        observations : `~gammapy.data.Observations`
            Container holding a list of `~gammapy.data.Observation`
        """
        if obs_id is None:
            obs_id = self.obs_table["OBS_ID"].data

        obs_list = []
        for _ in obs_id:
            try:
                obs = self.obs(_)
            except ValueError as err:
                if skip_missing:
                    log.warning(f"Skipping missing obs_id: {_!r}")
                    continue
                else:
                    raise err
            else:
                obs_list.append(obs)
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

        filename = outdir / self.DEFAULT_OBS_TABLE
        subobstable.write(str(filename), format="fits", overwrite=overwrite)

    def check(self, checks="all"):
        """Check index tables and data files.

        This is a generator that yields a list of dicts.
        """
        checker = DataStoreChecker(self)
        return checker.run(checks=checks)


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

    def __init__(self, paths):
        if isinstance(paths, (str, Path)):
            raise TypeError("Need list of paths, not a single string or Path object.")

        self.paths = [make_path(path) for path in paths]

        # Cache for EVENTS file header information, to avoid multiple reads
        self._events_info = {}

    def run(self):
        hdu_table = self.make_hdu_table()
        obs_table = self.make_obs_table()
        return DataStore(hdu_table=hdu_table, obs_table=obs_table)

    def get_events_info(self, path):
        if path not in self._events_info:
            self._events_info[path] = self.read_events_info(path)

        return self._events_info[path]

    def get_obs_info(self, path):
        # We could add or remove info here depending on what we want in the obs table
        return self.get_events_info(path)

    @staticmethod
    def read_events_info(path):
        log.debug(f"Reading {path}")
        with fits.open(path, memmap=False) as hdu_list:
            header = hdu_list["EVENTS"].header

        na_int, na_str = -1, "NOT AVAILABLE"

        info = {}
        # Note: for some reason `header["OBS_ID"]` is sometimes `str`, maybe trailing whitespace
        info["OBS_ID"] = int(header["OBS_ID"])
        info["RA_PNT"] = header["RA_PNT"]
        info["DEC_PNT"] = header["DEC_PNT"]
        pos = SkyCoord(info["RA_PNT"], info["DEC_PNT"], unit="deg").galactic
        info["GLON_PNT"] = pos.l.deg
        info["GLAT_PNT"] = pos.b.deg
        info["ZEN_PNT"] = 90 - float(header["ALT_PNT"])
        info["ALT_PNT"] = header["ALT_PNT"]
        info["AZ_PNT"] = header["AZ_PNT"]
        info["ONTIME"] = header["ONTIME"]
        info["LIVETIME"] = header["LIVETIME"]
        info["DEADC"] = header["DEADC"]
        info["TSTART"] = header["TSTART"]
        info["TSTOP"] = header["TSTOP"]
        info["DATE-OBS"] = header.get("DATE_OBS", na_str)
        info["TIME-OBS"] = header.get("TIME_OBS", na_str)
        info["DATE-END"] = header.get("DATE_END", na_str)
        info["TIME-END"] = header.get("TIME_END", na_str)
        info["N_TELS"] = header.get("N_TELS", na_int)
        info["OBJECT"] = header.get("OBJECT", na_str)

        # This is the info needed to link from EVENTS to IRFs
        info["TELESCOP"] = header.get("TELESCOP", na_str)
        info["CALDB"] = header.get("CALDB", na_str)
        info["IRF"] = header.get("IRF", na_str)

        # Not part of the spec, but good to know from which file the info comes
        info["EVENTS_FILENAME"] = str(path)
        info["EVENT_COUNT"] = header["NAXIS2"]

        # gti = Table.read(filename, hdu='GTI')
        # info['GTI_START'] = gti['START'][0]
        # info['GTI_STOP'] = gti['STOP'][0]

        return info

    def make_obs_table(self):
        rows = []
        for path in self.paths:
            row = self.get_obs_info(path)
            rows.append(row)

        names = list(rows[0].keys())
        table = ObservationTable(rows=rows, names=names)

        table["RA_PNT"].unit = "deg"
        table["DEC_PNT"].unit = "deg"
        table["GLON_PNT"].unit = "deg"
        table["GLAT_PNT"].unit = "deg"
        table["ZEN_PNT"].unit = "deg"
        table["ALT_PNT"].unit = "deg"
        table["AZ_PNT"].unit = "deg"
        table["ONTIME"].unit = "s"
        table["LIVETIME"].unit = "s"
        table["TSTART"].unit = "s"
        table["TSTOP"].unit = "s"

        # TODO: Values copied from one of the EVENTS headers
        # TODO: check consistency for all EVENTS files and handle inconsistent case
        # Transform times to first ref time? Or raise error for now?
        # Test by combining some HESS & CTA runs?
        m = table.meta
        m["MJDREFI"] = 51544
        m["MJDREFF"] = 5.0000000000e-01
        m["TIMEUNIT"] = "s"
        m["TIMESYS"] = "TT"
        m["TIMEREF"] = "LOCAL"

        m["HDUCLASS"] = "GADF"
        m["HDUDOC"] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
        m["HDUVERS"] = "0.2"
        m["HDUCLAS1"] = "INDEX"
        m["HDUCLAS2"] = "OBS"

        return table

    def make_hdu_table(self):
        rows = []
        for path in self.paths:
            rows.extend(self.get_hdu_table_rows(path))

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

    def get_hdu_table_rows(self, path):
        events_info = self.get_events_info(path)

        info = dict(
            OBS_ID=events_info["OBS_ID"],
            FILE_DIR=path.parent.as_posix(),
            FILE_NAME=path.name,
        )
        yield dict(HDU_TYPE="events", HDU_CLASS="events", HDU_NAME="EVENTS", **info)
        yield dict(HDU_TYPE="gti", HDU_CLASS="gti", HDU_NAME="GTI", **info)

        caldb_irf = CalDBIRF.from_meta(events_info)
        info = dict(
            OBS_ID=events_info["OBS_ID"],
            FILE_DIR=caldb_irf.file_dir,
            FILE_NAME=caldb_irf.file_name,
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
    def file_name(self):
        return "irf_file.fits"
