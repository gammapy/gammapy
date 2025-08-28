# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
from astropy.io import fits
from astropy.table import Table
from gammapy.data import EventListMetaData, EventList, ObservationTable
from gammapy.utils.scripts import make_path
from gammapy.utils.metadata import CreatorMetaData


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
                    return ObservationTable.from_gadf02_table(table_disk)
                elif version == "0.3":
                    return ObservationTable.from_gadf02_table(table_disk)
                else:
                    raise ValueError(f"Unknown version :{version}")
            else:
                raise ValueError(f"Unknown format :{format}")


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
