# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
from astropy.io import fits
from astropy.table import Table
from gammapy.data import EventListMetaData, EventList, ObservationTable
from gammapy.utils.scripts import make_path
from gammapy.utils.metadata import CreatorMetaData
from gammapy.data.metadata import METADATA_FITS_KEYS, ObservationMetaData
from gammapy.utils.time import time_ref_from_dict
from astropy.units import Quantity


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

    def __init__(self, hdu="OBS_INDEX", checksum=False):
        self.hdu = hdu
        self.checksum = checksum

    @staticmethod
    def identify_format_from_hdu(obs_hdu):
        """Identify format for HDU header keywords."""
        hduclass = obs_hdu.header.get("HDUCLASS", "unknown")
        hduvers = obs_hdu.header.get("HDUVERS", "unknown")
        return [hduclass.lower(), hduvers.lower()]

    def read(self, filename, format="gadf0.3"):
        """Read ObservationTable from file.

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
            obs_hdu = hdulist[self.hdu]

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
                    return self.from_gadf03_hdu(obs_hdu)
                else:
                    raise ValueError(f"Unknown version :{version}")
            else:
                raise ValueError(f"Unknown format :{format}")

    @staticmethod
    def from_gadf02_hdu(obs_hdu):
        """Create ObservationTable from gadf0.2 HDU."""
        table_disk = Table.read(obs_hdu)  # table_disk !
        meta = ObservationMetaData.from_header(table_disk.meta)  # TEST
        print(meta)  # TEST

        # Required names to fill internal table format, for GADF v.0.2, will be extended after checking POINTING. Similar to PR#5954.
        required_names_on_disk = [
            "OBS_ID",
            "OBJECT",
            "GEOLON",
            "GEOLAT",
            "ALTITUDE",
            "TSTART",
            "TSTOP",
        ]

        # Get colnames of disk_table
        names_disk = table_disk.colnames

        # Get header of obs-index table.
        meta = table_disk.meta

        # Check which info is given for POINTING
        # Used commit 16ce9840f38bea55982d2cd986daa08a3088b434 by @registerrier
        if "OBS_MODE" in names_disk:
            # like in data_store.py:
            if table_disk["OBS_MODE"] == "DRIFT":
                required_names_on_disk.append("ALT_PNT")
                required_names_on_disk.append("AZ_PNT")
            else:
                required_names_on_disk.append("RA_PNT")
                required_names_on_disk.append("DEC_PNT")
        else:
            # if "OBS_MODE" not given, decide based on what is given, RADEC or ALTAZ
            if "RA_PNT" in names_disk:
                required_names_on_disk.append("RA_PNT")
                required_names_on_disk.append("DEC_PNT")
            elif "ALT_PNT" in names_disk:
                required_names_on_disk.append("ALT_PNT")
                required_names_on_disk.append("AZ_PNT")
            else:
                raise RuntimeError("Neither RADEC nor ALTAZ is given in table on disk!")

        # Used: aeb1ea01e60e1f02c5fb59f50141c81e0b2fb8f6:
        missing_names = set(required_names_on_disk).difference(
            names_disk + list(meta.keys())
        )
        if len(missing_names) != 0:
            raise RuntimeError(
                f"Not all columns required for GADF v.0.2 were found in file. Missing: {missing_names}"
            )
        #             )  # looked into gammapy/workflow/core.py

        # Create internal table "table_internal" with all names, corresp. units, types and descriptions, for the internal table model.
        table_internal = ObservationTable(ObservationTable._reference_table())

        # Fill internal table for mandatory columns by constructing the table row-wise with the internal representations.
        number_of_observations = len(
            table_disk
        )  # Get number of observations, equal to number of rows in table on disk.

        for i in range(number_of_observations):
            row_internal = [str(table_disk[i]["OBS_ID"]), str(table_disk[i]["OBJECT"])]

            if "RA_PNT" in required_names_on_disk:
                row_internal.append(
                    METADATA_FITS_KEYS["pointing"]["radec_mean"]["input"](
                        {
                            "RA_PNT": table_disk[i]["RA_PNT"],
                            "DEC_PNT": table_disk[i]["DEC_PNT"],
                        }
                    )
                )
            elif "ALT_PNT" in required_names_on_disk:
                row_internal.append(
                    METADATA_FITS_KEYS["pointing"]["altaz_mean"]["input"](
                        {
                            "ALT_PNT": table_disk[i]["ALT_PNT"],
                            "AZ_PNT": table_disk[i]["AZ_PNT"],
                        }
                    )
                )

            row_internal.append(
                METADATA_FITS_KEYS["observation"]["location"]["input"](
                    {
                        "GEOLON": meta["GEOLON"],
                        "GEOLAT": meta["GEOLAT"],
                        "ALTITUDE": meta["ALTITUDE"],
                    }
                )
            )

            # from @properties "time_ref", "time_start", "time_stop"
            time_ref = time_ref_from_dict(meta)
            if "TIMEUNIT" in meta.keys():
                time_unit = meta["TIMEUNIT"]
            else:
                time_unit = "s"
            row_internal.append(
                time_ref
                + Quantity(table_disk[i]["TSTART"].astype("float64"), time_unit)
            )
            row_internal.append(
                time_ref + Quantity(table_disk[i]["TSTOP"].astype("float64"), time_unit)
            )

            # )  # like in event_list.py, l.201, commit: 08c6f6a
            table_internal.add_row(
                row_internal
            )  # Add row to internal table (fill table).

        # Load optional columns, whose names are not already processed, automatically into internal table.
        opt_names = set(names_disk).difference(required_names_on_disk)
        for name in opt_names:  # add column-wise all optional column-data present in file, independent of format.
            table_internal[name] = table_disk[name]

        table_internal.meta = meta
        # return internal table, instead of copy of disk-table like before.
        return table_internal
        # return ObservationTable(table=table, meta=meta)

    @staticmethod
    def from_gadf03_hdu(obs_hdu):
        """Create ObservationTable from gadf0.3 HDU."""
        table_disk = Table.read(obs_hdu)  # table_disk !
        # meta = ObservationMetaData.from_header(table.meta)
        # print(meta)
        return table_disk
        # return ObservationTable(table=table, meta=meta)


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
