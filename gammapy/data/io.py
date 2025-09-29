# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
import logging
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astropy.units import Quantity
from gammapy.data import EventListMetaData, EventList, ObservationTable
from gammapy.utils.scripts import make_path
from gammapy.utils.metadata import CreatorMetaData
from gammapy.utils.time import time_ref_from_dict

log = logging.getLogger(__name__)


class EventListReader:
    """Reader class for EventList.

    Format specification: :ref:`gadf:iact-events`

    Parameters
    ----------
    hdu : str
        Name of events HDU. Default is "EVENTS".
    checksum : bool
        If True checks both DATASUM and CHECKSUM cards in the file headers. Default is False.
    """

    def __init__(self, hdu="EVENTS", checksum=False):
        self.hdu = hdu
        self.checksum = checksum

    @staticmethod
    def from_gadf_hdu(events_hdu):
        """Create EventList from gadf HDU."""
        table = Table.read(events_hdu)
        meta = EventListMetaData.from_header(table.meta)

        # This is not a strict check on input. It just checks that required information is there.
        required_colnames = set(["RA", "DEC", "TIME", "ENERGY"])
        if not required_colnames.issubset(set(table.colnames)):
            missing_columns = required_colnames.difference(set(table.colnames))
            raise ValueError(
                f"GADF event table does not contain required columns {missing_columns}"
            )

        met = u.Quantity(table["TIME"].astype("float64"), "second")
        time = time_ref_from_dict(table.meta) + met

        energy = table["ENERGY"].quantity

        ra = table["RA"].quantity
        dec = table["DEC"].quantity

        removed_colnames = ["RA", "DEC", "GLON", "GLAT", "TIME", "ENERGY"]

        new_table = Table(
            {"TIME": time, "ENERGY": energy, "RA": ra, "DEC": dec}, meta=table.meta
        )
        for name in table.colnames:
            if name not in removed_colnames:
                new_table.add_column(table[name])

        return EventList(new_table, meta)

    @staticmethod
    def identify_format_from_hduclass(events_hdu):
        """Identify format for HDU header keywords."""
        hduclass = events_hdu.header.get("HDUCLASS", "unknown")
        return hduclass.lower()

    def read(self, filename, format="gadf"):
        """Read EventList from file.

        Parameters
        ----------
        filename : `pathlib.Path`, str
            Filename
        format : {"gadf"}, optional
            format of the EventList. Default is 'gadf'.
            If None, will try to guess from header.
        """
        filename = make_path(filename)

        with fits.open(filename) as hdulist:
            events_hdu = hdulist[self.hdu]

            if self.checksum:
                if events_hdu.verify_checksum() != 1:
                    warnings.warn(
                        f"Checksum verification failed for HDU {self.hdu} of {filename}.",
                        UserWarning,
                    )

            if format is None:
                format = self.identify_format_from_hduclass(events_hdu)

            if format == "gadf" or format == "ogip":
                return self.from_gadf_hdu(events_hdu)
            else:
                raise ValueError(f"Unknown format :{format}")


class EventListWriter:
    """Writer class for EventList."""

    def __init__(self):
        pass

    @staticmethod
    def _to_gadf_table_hdu(event_list):
        """Convert input event list to a `~astropy.io.fits.BinTableHDU` according gadf."""
        gadf_table = event_list.table.copy()
        gadf_table.remove_column("TIME")

        reference_time = time_ref_from_dict(gadf_table.meta)
        gadf_table["TIME"] = (event_list.time - reference_time).to("s")

        bin_table = fits.BinTableHDU(gadf_table, name="EVENTS")

        # A priori don't change creator information
        if event_list.meta.creation is None:
            event_list.meta.creation = CreatorMetaData()
        else:
            event_list.meta.creation.update_time()

        bin_table.header.update(event_list.meta.to_header())
        return bin_table

    def to_hdu(self, event_list, format="gadf"):
        """
        Convert input event list to a `~astropy.io.fits.BinTableHDU` according to format.

        Parameters
        ----------
        format : str, optional
            Output format, currently only "gadf" is supported. Default is "gadf".

        Returns
        -------
        hdu : `astropy.io.fits.BinTableHDU`
            EventList converted to FITS representation.
        """
        if format != "gadf":
            raise ValueError(f"Only the 'gadf' format supported, got {format}")

        return self._to_gadf_table_hdu(event_list)


class ObservationTableReader:
    """Reader class for ObservationTable"""

    def read(self, filename, hdu=None):
        """Read `~gammapy.data.ObservationTable` from file.

        Parameters
        ----------
        filename : `pathlib.Path`, str
            Filename
        hdu : str, optional
            Name of observation table HDU. Default is None.

        Returns
        -------
        ObservationTable : `~gammapy.data.ObservationTable`
            ObservationTable from disk converted into internal data format.
        """
        filename = make_path(filename)

        table_disk = Table.read(filename, format="fits", hdu=hdu)

        format = table_disk.meta.get("HDUCLASS")

        if format is None:
            format = "GADF"
            log.warning(
                f"Could not infer fileformat from metadata in {filename}, assuming GADF."
            )

        if format.upper() in ["GADF", "OGIP"]:
            return self._from_gadf_table(table_disk)
        else:
            raise ValueError(f"Unknown fileformat :{format}")

    @staticmethod
    def _from_gadf_table(table_gadf):
        """Convert gadf observation table into`~gammapy.data.ObservationTable`.

        Parameters
        ----------
        table_gadf : `~astropy.Table.table`
            Table in gadf 0.2/0.3 format, see [1], [2].

        Returns
        -------
        ObservationTable : `~gammapy.data.ObservationTable`
            ObservationTable in internal data format.

        References
        ----------
        [1] Gamma-ray astronomy community, 2018, 'Data formats for gamma-ray astronomy v0.2 <https://gamma-astro-data-formats.readthedocs.io/en/v0.2/data_storage/obs_index/index.html>'_
        [2] Gamma-ray astronomy community, 2018, 'Data formats for gamma-ray astronomy v0.3 <https://gamma-astro-data-formats.readthedocs.io/en/v0.3/data_storage/obs_index/index.html>'_
        """

        names_gadf = table_gadf.colnames
        meta_gadf = table_gadf.meta

        required_names_gadf = [
            "OBS_ID",
        ]

        missing_names = set(required_names_gadf).difference(names_gadf)
        if len(missing_names) != 0:
            raise RuntimeError(
                f"Not all columns required to read from GADF were found in file. Missing: {missing_names}"
            )

        time_columns = set(["TSTART", "TSTOP"]).intersection(set(names_gadf))

        if len(time_columns) != 0:
            try:
                time_ref = time_ref_from_dict(meta_gadf)
                time_unit = meta_gadf["TIMEUNIT"]
                for colname in time_columns:
                    try:
                        time_object = time_ref + Quantity(
                            table_gadf[colname].astype("float64"), time_unit
                        )
                        table_gadf[colname] = time_object
                    except ValueError:
                        warnings.warn(f"Invalid unit for column {colname}.")
                        table_gadf.remove_column(colname)
                    except TypeError:
                        warnings.warn(f"Could not convert type for column {colname}.")
                        table_gadf.remove_column(colname)
            except KeyError:
                warnings.warn(
                    "Found column TSTART or TSTOP in gadf table, but can not create columns in internal format (MixinColumn Time) due to missing header keywords in file."
                )
                table_gadf.remove_columns(time_columns)

        return ObservationTable(
            data=table_gadf, meta=meta_gadf
        )  # TODO: Replace "data" by "table" once new constructor is added in class ObservationTable.
