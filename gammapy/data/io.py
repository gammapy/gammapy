# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
from astropy.io import fits
from astropy.table import Table
from gammapy.data import EventListMetaData, EventList
from gammapy.utils.scripts import make_path


class EventListReader:
    """Reader class for EventList.

    Format specification: :ref:`gadf:iact-events`

    Parameters
    ----------
    filename : `pathlib.Path`, str
        Filename
    hdu : str
        Name of events HDU. Default is "EVENTS".
    checksum : bool
        If True checks both DATASUM and CHECKSUM cards in the file headers. Default is False.
    """

    def __init__(self, filename, hdu="EVENTS", checksum=False):
        self.filename = make_path(filename)
        self.hdu = hdu
        self.checksum = checksum

    @staticmethod
    def from_gadf_hdu(events_hdu):
        """Create EventList from gadf HDU."""
        table = Table.read(events_hdu)
        meta = EventListMetaData.from_header(table.meta)
        return EventList(table=table, meta=meta)

    def read(self):
        filename = self.filename

        with fits.open(filename) as hdulist:
            events_hdu = hdulist[self.hdu]
            if self.checksum:
                if events_hdu.verify_checksum() != 1:
                    warnings.warn(
                        f"Checksum verification failed for HDU {self.hdu} of {filename}.",
                        UserWarning,
                    )

            hduclass = events_hdu.header.get("HDUCLASS", "GADF")
            if hduclass.upper() == "GADF":
                return self.from_gadf_hdu(events_hdu)
            if hduclass.upper() == "OGIP":
                # Fermi event list for now use same function
                return self.from_gadf_hdu(events_hdu)
            else:
                raise ValueError(f"Unknown HDUCLASS :{hduclass}")
