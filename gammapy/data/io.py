# Licensed under a 3-clause BSD style license - see LICENSE.rst
from typing import Optional
from astropy.io import fits
from astropy.table import Table
from pydantic import BaseModel, ValidationError, validator
from gammapy.utils.scripts import make_path
from gammapy.utils.table_validator import GADF_EVENT_TABLE_DEFINITION, TableValidator


class GADFEventsHeader(BaseModel):
    OBS_ID: int
    TSTART: float
    TSTOP: float
    ONTIME: float
    LIVETIME: float
    DEADC: float
    OBS_MODE: str
    RA_PNT: float
    DEC_PNT: float
    ALT_PNT: float
    AZ_PNT: float
    EQUINOX: float
    RADECSYS: str
    ORIGIN: str
    TELESCOP: str
    INSTRUME: str
    CREATOR: str
    HDUCLASS: str = "GADF"
    HDUDOC: str = "https://gamma-astro-data-formats.readthedocs.io/en/v0.3/index.html"
    HDUVERS: str = "v0.3"
    HDUCLAS1: str = "EVENTS"

    OBSERVER: Optional[str]
    CREATED: Optional[str]
    OBJECT: Optional[str]
    RA_OBJ: Optional[float]
    DEC_OBJ: Optional[float]
    EV_CLASS: Optional[str]
    TELAPSE: Optional[float]
    TELLIST: Optional[str]
    N_TELS: Optional[int]
    TASSIGN: Optional[str]
    DST_VER: Optional[str]
    ANA_VER: Optional[str]
    CAL_VER: Optional[str]
    CONV_DEP: Optional[float]
    CONV_RA: Optional[float]
    CONV_DEC: Optional[float]
    TRGRATE: Optional[float]
    ZTRGRATE: Optional[float]
    MUONEFF: Optional[float]
    BROKPIX: Optional[float]
    AIRTEMP: Optional[float]
    PRESSURE: Optional[float]
    RELHUM: Optional[float]
    NSBLEVEL: Optional[float]

    @validator("HDUCLAS1")
    def _validate_hduclas1(cls, v):
        if v == "EVENTS":
            return v
        else:
            raise ValidationError(
                f"Incorrect HDUCLAS1. Expected EVENTS got {v} instead."
            )

    def __getitem__(self, name):
        if hasattr(self, name):
            return self.__getattribute__(name)
        else:
            raise KeyError

    def to_header(self):
        hdr_dict = {}
        for key, item in self.dict().items():
            hdr_dict[key.upper()] = item.__str__()
        return hdr_dict

    @classmethod
    def from_header(cls, hdr):
        kwargs = {}
        for key in cls.__fields__.keys():
            kwargs[key] = hdr.get(key.upper(), None)
        return cls(**kwargs)


class GADFEventsReaderWriter:
    """IO class for GADF events."""

    table_validator = TableValidator.from_builtin(GADF_EVENT_TABLE_DEFINITION)
    meta_validator = GADFEventsHeader

    @classmethod
    def read(cls, filename, hdu="EVENTS"):
        filename = make_path(filename)
        table = Table.read(filename, hdu)

        hdr = cls.meta_validator.from_header(table.meta)
        table = cls.table_validator.run(table)
        return table, hdr

    def to_table_hdu(self):
        """Export to table HDU."""
        table_hdu = fits.BinTableHDU(self.table, name="EVENTS")
        table_hdu.header.update(self.header.to_header())
        return table_hdu

    @classmethod
    def from_eventlist(cls, eventlist):
        header = eventlist.table.meta
        table = eventlist.table
        return cls(header=header, table=table)
