# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
from astropy.io import fits
from astropy.table import Table
from astropy import table
from astropy import units as u
from astropy.units import Quantity
from gammapy.data import EventListMetaData, EventList, ObservationTable
from gammapy.utils.scripts import make_path
from gammapy.utils.metadata import CreatorMetaData
from gammapy.utils.time import time_ref_from_dict


class ObservationTableReader:
    """Reader class for ObservationTable"""

    def read(self, filename, hdu=None):
        """Read ObservationTable from file.

        Parameters
        ----------
        filename : `pathlib.Path`, str
            Filename
        hdu : str, optional
            Name of observation table HDU. Default is None.
        """
        filename = make_path(filename)

        table_disk = Table.read(filename, format="fits", hdu=hdu)
        table_disk_meta = table_disk.meta

        format = table_disk_meta.get("HDUCLASS", "unknown")
        version = table_disk_meta.get("HDUVERS", "unknown")

        if (format == "unknown") or (format is None):
            format = "GADF"
            version = "0.3"
            warnings.warn(
                UserWarning(
                    f"Could not infer fileformat from metadata in {filename}, assuming GADF."
                )
            )

        if format == "GADF" or format == "OGIP":
            if version == "0.2" or version == "0.3":
                return self.from_gadf_table(table_disk)
            else:
                raise ValueError(f"Unknown {format} version :{version}")
        else:
            raise ValueError(f"Unknown fileformat :{format}")

    @staticmethod
    def from_gadf_table(table_gadf):
        """Convert gadf observation table into internal table model.
        https://gamma-astro-data-formats.readthedocs.io/en/v0.3/data_storage/obs_index/index.html

        Parameters
        ----------
        table_gadf : `~astropy.Table.table`
            Table in gadf 0.2/0.3 format.

        Returns
        -------
        ObservationTable : `~gammapy.data.ObservationTable`
            ObservationTable in internal data format.
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

        removed_names = []

        try:
            obs_id = table_gadf["OBS_ID"].astype("int")
        except ValueError:
            raise RuntimeError(
                "Could not convert OBS_ID to int. Can not create table without OBS_ID."
            )
        new_table = Table({"OBS_ID": obs_id}, meta=meta_gadf)
        new_table = table.unique(new_table, keys="OBS_ID")
        removed_names.append("OBS_ID")

        for colname in ["RA_PNT", "DEC_PNT", "ALT_PNT", "AZ_PNT"]:
            if colname in names_gadf:
                try:
                    new_table[colname] = Quantity(
                        table_gadf[colname].astype("float64"),
                        u.Unit(table_gadf[colname].unit),
                    ).to(u.deg)
                except TypeError:
                    warnings.warn(f"Could not convert unit for column {colname}.")
                removed_names.append(colname)

        for colname in ["TSTART", "TSTOP"]:
            if colname in names_gadf:
                try:
                    time_ref = time_ref_from_dict(meta_gadf)
                    time_unit = meta_gadf["TIMEUNIT"]
                except KeyError:
                    warnings.warn(
                        "Found column TSTART or TSTOP in gadf table, but can not create columns in internal format (MixinColumn Time) due to missing header keywords in file."
                    )
                    removed_names.append(colname)

                if colname not in removed_names:
                    try:
                        time_object = time_ref + Quantity(
                            table_gadf[colname].astype("float64"), time_unit
                        )
                        new_table[colname] = time_object
                    except TypeError:
                        warnings.warn(
                            f"Could not build time object for column {colname}."
                        )
                    removed_names.append(colname)

        for name in names_gadf:
            if name not in removed_names:
                new_table.add_column(table_gadf[name])

        return ObservationTable(table=new_table, meta=meta_gadf)


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
        return EventList(table=table, meta=meta)

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
        bin_table = fits.BinTableHDU(event_list.table, name="EVENTS")

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
