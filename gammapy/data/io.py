# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.units import Quantity
from gammapy.data import EventListMetaData, EventList, ObservationTable
from gammapy.utils.scripts import make_path
from gammapy.utils.metadata import CreatorMetaData
from gammapy.utils.types import cast_func
from gammapy.utils.time import time_ref_from_dict


class ObservationTableReader:
    """Reader class for ObservationTable"""

    """Based on class EventListReader!!!
    
    
    Parameters
    ----------
    hdu : str
        Name of observation table HDU. Default is "OBS_INDEX".
    checksum : bool
        If True checks both DATASUM and CHECKSUM cards in the file headers. Default is False.
    """

    def __init__(self, checksum=False):
        self.checksum = checksum

    @staticmethod
    def identify_format_from_hdu(obs_hdu):
        """Identify format for HDU header keywords."""
        hduclass = obs_hdu.header.get("HDUCLASS", "unknown")
        hduvers = obs_hdu.header.get("HDUVERS", "unknown")
        return [hduclass.lower(), hduvers.lower()]

    def read(self, filename, format="gadf0.3", hdu="OBS_INDEX"):
        """Read ObservationTable from file.
        For now, only gadf 0.2 reader implemented and called for both gadf 0.2 and gadf 0.3.

        Parameters
        ----------
        filename : `pathlib.Path`, str
            Filename
        format : {"gadf0.2/0.3"}, optional
            format and its version, of the ObservationTable. Default is 'gadf0.3'.
            If None, will try to guess from header.
        """
        filename = make_path(filename)

        with fits.open(filename) as hdulist:
            # If hdu extension not found, assume obs-index hdu at 1 and raise warning.
            hdu_names = []
            for hduobject in hdulist:
                hdu_names.append(hduobject.name)
            if hdu in hdu_names:
                obs_hdu = hdulist[hdu]
            else:
                obs_hdu = hdulist[1]
                warnings.warn(
                    f"Extension {hdu} was not found in file, assuming obs-index HDU at index 1.",
                    UserWarning,
                )

            if self.checksum:
                if obs_hdu.verify_checksum() != 1:
                    warnings.warn(
                        f"Checksum verification failed for HDU {self.hdu} of {filename}.",
                        UserWarning,
                    )
            # Read Table as is on disk.
            table_disk = Table.read(obs_hdu)

            # Convert table on disk, by calling converter for specific format.
            if format is None:
                formatname = self.identify_format_from_hdu(obs_hdu)[0]
                version = self.identify_format_from_hdu(obs_hdu)[1]
            else:
                formatname = format[0:4]
                version = format[4:]

            if formatname == "gadf" or formatname == "ogip":
                if version == "0.2":
                    return self.from_gadf02_table(table_disk)
                elif version == "0.3":
                    return self.from_gadf02_table(table_disk)
                else:
                    raise ValueError(f"Unknown version :{version}")
            else:
                raise ValueError(f"Unknown format :{format}")

    @staticmethod
    def from_gadf02_table(table_gadf):
        """Convert gadf 0.2 observation table into internal table model.

        This function is called by the read-method if the format is gadf 0.2,
        but also used for converting constructed observation-tables, like in
        'gammapy/data/data_store.py' or 'gammapy/data/tests/test_obs_table.py'.

        Parameters
        ----------
        table_gadf : `~astropy.Table.table`
            Table in gadf 0.2 format.

        Returns
        -------
        ObservationTable : `~gammapy.data.ObservationTable`
            ObservationTable in internal data format.
        """

        names_gadf = table_gadf.colnames
        meta_gadf = table_gadf.meta

        # Required names in gadf 0.2 table, in order to fill internal table format.
        # Requirement is weak for conversion from gadf to internal.
        # See: https://gamma-astro-data-formats.readthedocs.io/en/v0.2/data_storage/obs_index/index.html#required-columns
        required_names_gadf = [
            "OBS_ID",
        ]
        # Used: aeb1ea01e60e1f02c5fb59f50141c81e0b2fb8f6:
        missing_names = set(required_names_gadf).difference(
            names_gadf + list(meta_gadf.keys())
        )
        if len(missing_names) != 0:
            raise RuntimeError(
                f"Not all columns required to read from GADF v.0.2 were found in file. Missing: {missing_names}"
            )  # looked into gammapy/workflow/core.py

        # Names to be removed, to handle optional columns.
        removed_names = []

        # Convert gadf data by ensuring correct types and units, as well as Time-Object for times
        # for internal model representation, in case data corresponding to it is given.

        # Create new table with mandatory column OBS_ID.
        obs_id = cast_func(table_gadf["OBS_ID"], np.dtype(int))
        new_table = Table({"OBS_ID": obs_id}, meta=meta_gadf)
        removed_names.append("OBS_ID")

        if "RA_PNT" in names_gadf:
            ra_pnt = cast_func(table_gadf["RA_PNT"], np.dtype(float))
            new_table["RA_PNT"] = ra_pnt * u.deg
            removed_names.append("RA_PNT")
        if "DEC_PNT" in names_gadf:
            dec_pnt = cast_func(table_gadf["DEC_PNT"], np.dtype(float))
            new_table["DEC_PNT"] = dec_pnt * u.deg
            removed_names.append("DEC_PNT")

        # Used commit 16ce9840f38bea55982d2cd986daa08a3088b434 by @registerrier

        # Used here code from the @properties: "time_ref", "time_start", "time_stop"
        # and code for ref-time calculation from commit: 08c6f6a for event_list.py.
        if "TSTART" in names_gadf or "TSTOP" in names_gadf:
            if (
                "MJDREFI" in meta_gadf.keys()
                and "MJDREFF" in meta_gadf.keys()
                and "TIMESYS" in meta_gadf.keys()
            ):  # Choice to be mandatory to construct meaningful time object!
                if "TIMEUNIT" in meta_gadf.keys():
                    time_unit = meta_gadf["TIMEUNIT"]
                else:
                    time_unit = "s"
                time_ref = time_ref_from_dict(meta_gadf)
                if "TSTART" in names_gadf:
                    tstart = time_ref + Quantity(
                        table_gadf["TSTART"].astype("float64"), time_unit
                    )
                    new_table["TSTART"] = tstart
                    removed_names.append("TSTART")
                if "TSTOP" in names_gadf:
                    tstop = time_ref + Quantity(
                        table_gadf["TSTOP"].astype("float64"), time_unit
                    )
                    new_table["TSTOP"] = tstop
                    removed_names.append("TSTOP")
            else:
                raise RuntimeError(
                    "Found column TSTART or TSTOP in gadf 0.2 table, but its metadata does not contain mandatory keyword(s) to calculate reference time for conversion to internal model."
                )

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
