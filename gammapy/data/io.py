# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
from astropy.io import fits
from astropy.table import Table
from gammapy.data import EventListMetaData, EventList, ObservationTable
from gammapy.utils.scripts import make_path
from gammapy.utils.metadata import CreatorMetaData
from gammapy.utils.time import time_ref_from_dict
from gammapy.utils.fits import (
    skycoord_from_dict,
    earth_location_to_dict,
)
from astropy.units import Quantity
from gammapy.data import observatory_locations


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

    def read(self, filename, format="gadf0.3", hdu=None):
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
            # In case of hdu specification in kwarg, use it, otherwise assume, HDU has only obs-index-table.
            if hdu is not None:
                obs_hdu = hdulist[hdu]
            else:
                obs_hdu = hdulist

            if self.checksum:
                if obs_hdu.verify_checksum() != 1:
                    warnings.warn(
                        f"Checksum verification failed for HDU {self.hdu} of {filename}.",
                        UserWarning,
                    )

            if format is None:
                formatname = self.identify_format_from_hdu(obs_hdu)[0]
                version = self.identify_format_from_hdu(obs_hdu)[1]
            else:
                formatname = format[0:4]
                version = format[4:]

            if formatname == "gadf" or formatname == "ogip":
                if version == "0.2":
                    return self.from_gadf02_hdu(obs_hdu)
                elif version == "0.3":
                    return self.from_gadf02_hdu(obs_hdu)
                else:
                    raise ValueError(f"Unknown version :{version}")
            else:
                raise ValueError(f"Unknown format :{format}")

    @staticmethod
    def from_gadf02_hdu(obs_hdu):
        """Create ObservationTable from gadf0.2 HDU."""
        table_disk = Table.read(obs_hdu)
        meta = table_disk.meta

        names_disk = table_disk.colnames

        # Mandatory names to fill internal table format from GADF v.0.2, will be extended after checking POINTING.
        # Subset of: https://gamma-astro-data-formats.readthedocs.io/en/v0.2/data_storage/obs_index/index.html#required-columns
        required_names_on_disk = [
            "OBS_ID",
        ]
        removed_names = []
        # https://stackoverflow.com/questions/74412503/cannot-access-local-variable-a-where-it-is-not-associated-with-a-value-but used for debugging
        # location = earth_location_from_dict(
        #     {
        #         "GEOLON": float(meta["GEOLON"]),
        #         "GEOLAT": float(meta["GEOLAT"]),
        #         "ALTITUDE": float(meta["ALTITUDE"]),
        #     }
        # )

        # Create new table with mandatory column OBS_ID.
        obs_id = table_disk["OBS_ID"]
        new_table = Table({"OBS_ID": obs_id}, meta=meta)
        removed_names.append("OBS_ID")

        # If observatory location is given, try to find instrument name and add to meta.
        meta["INSTRUME"] = "UNKNOWN"  # if not found, UNKNOWN.
        if "GEOLON" in meta.keys():
            for instrument in observatory_locations.keys():
                # Ideally compare if observatory_locations[instrument] == location.
                loc = earth_location_to_dict(observatory_locations[instrument])
                tol = 1e-5
                if float(meta["GEOLON"]) < loc["GEOLON"] * 1 + tol:
                    if float(meta["GEOLON"]) > loc["GEOLON"] * 1 - tol:
                        meta["INSTRUME"] = instrument
                        break

        # Check which info is given for POINTING
        # Used commit 16ce9840f38bea55982d2cd986daa08a3088b434 by @registerrier

        # Assume observation is pointed.
        pointing = True  # Assumption

        # Check if DRIFT Mode.
        if "OBS_MODE" in names_disk:
            # like in data_store.py:
            if table_disk["OBS_MODE"] in ["DRIFT", "WOBBLE", "RASTER", "SLEW", "SCAN"]:
                pointing = False

        # For pointed observations construct POINTING:
        if pointing is True:
            if "RA_PNT" in names_disk and "DEC_PNT" in names_disk:
                pointing = skycoord_from_dict(
                    {
                        "RA_PNT": table_disk["RA_PNT"],
                        "DEC_PNT": table_disk["DEC_PNT"],
                    },
                    frame="icrs",
                    ext="PNT",
                )
                removed_names.append("RA_PNT")
                removed_names.append("DEC_PNT")
                new_table["POINTING"] = pointing
            else:
                raise RuntimeError(
                    "RA_PNT and DEC_PNT not found in gadf file, but needed for POINTING."
                )

        # Used: aeb1ea01e60e1f02c5fb59f50141c81e0b2fb8f6:
        missing_names = set(required_names_on_disk).difference(
            names_disk + list(meta.keys())
        )
        if len(missing_names) != 0:
            raise RuntimeError(
                f"Not all columns required to read from GADF v.0.2 were found in file. Missing: {missing_names}"
            )
        # looked into gammapy/workflow/core.py

        # elif "ALT_PNT" in required_names_on_disk:
        #     pointing = skycoord_from_dict(
        #         {
        #             "ALT_PNT": table_disk["ALT_PNT"],
        #             "AZ_PNT": table_disk["AZ_PNT"],
        #         },
        #         frame="altaz",
        #         ext="PNT",
        #     )
        #     removed_names.append("ALT_PNT")
        #     removed_names.append("AZ_PNT")
        #     new_table["POINTING"] = pointing

        # from @properties "time_ref", "time_start", "time_stop"
        if "TSTART" in names_disk or "TSTOP" in names_disk:
            if "MJDREFI" in meta:  # mandatory!!!
                if "TIMEUNIT" in meta.keys():
                    time_unit = meta["TIMEUNIT"]
                else:
                    time_unit = "s"
                time_ref = time_ref_from_dict(meta)
                if "TSTART" in names_disk:
                    tstart = time_ref + Quantity(
                        table_disk["TSTART"].astype("float64"), time_unit
                    )
                    new_table["TSTART"] = tstart
                    removed_names.append("TSTART")
                if "TSTOP" in names_disk:
                    tstop = time_ref + Quantity(
                        table_disk["TSTOP"].astype("float64"), time_unit
                    )
                new_table["TSTOP"] = tstop
                removed_names.append("TSTOP")
            else:
                raise RuntimeError(
                    "Metadata of table on disk does not contain mandatory keywords to calculate reference time."
                )

        # like in event_list.py, l.201, commit: 08c6f6a

        for name in names_disk:
            if name not in removed_names:
                new_table.add_column(table_disk[name])

        return ObservationTable(table=new_table, meta=meta)


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
