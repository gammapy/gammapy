# Licensed under a 3-clause BSD style license - see LICENSE.rst
from typing import Optional
from astropy.io import fits
from astropy.table import Table
from pydantic import BaseModel, ValidationError, validator
from gammapy.utils.scripts import make_path


class ColumnType(BaseModel):
    dtype: str
    unit: Optional[str]
    required: bool = False

    def validate_column(self, column):
        if column.unit is None:
            if self.unit is None or self.unit == "":
                return column
        elif column.unit.is_equivalent(self.unit):
            return column
        else:
            raise ValidationError


class GADFTableValidator(BaseModel):
    EVENT_ID: ColumnType = ColumnType(dtype="int", required=True)
    TIME: ColumnType = ColumnType(dtype="float64", unit="s", required=True)
    RA: ColumnType = ColumnType(dtype="float64", unit="deg", required=True)
    DEC: ColumnType = ColumnType(dtype="float64", unit="deg", required=True)
    ENERGY: ColumnType = ColumnType(dtype="float64", unit="TeV", required=True)

    EVENT_TYPE: ColumnType = ColumnType(dtype="int8")
    MULTIP: ColumnType = ColumnType(dtype="int")
    GLON: ColumnType = ColumnType(dtype="float", unit="deg")
    GLAT: ColumnType = ColumnType(dtype="float", unit="deg")
    ALT: ColumnType = ColumnType(dtype="float", unit="deg")
    AZ: ColumnType = ColumnType(dtype="float", unit="deg")
    DETX: ColumnType = ColumnType(dtype="float", unit="deg")
    DETY: ColumnType = ColumnType(dtype="float", unit="deg")
    THETA: ColumnType = ColumnType(dtype="float", unit="deg")
    PHI: ColumnType = ColumnType(dtype="float", unit="deg")
    GAMMANESS: ColumnType = ColumnType(dtype="float")
    DIR_ERR: ColumnType = ColumnType(dtype="float", unit="deg")
    ENERGY_ERR: ColumnType = ColumnType(dtype="float", unit="TeV")
    COREX: ColumnType = ColumnType(dtype="float", unit="m")
    COREY: ColumnType = ColumnType(dtype="float", unit="m")
    CORE_ERR: ColumnType = ColumnType(dtype="float", unit="m")
    XMAX: ColumnType = ColumnType(dtype="float", unit="")
    XMAX_ERR: ColumnType = ColumnType(dtype="float", unit="")
    HIL_MSW: ColumnType = ColumnType(dtype="float", unit="")
    HIL_MSW_ERR: ColumnType = ColumnType(dtype="float", unit="")
    HIL_MSL: ColumnType = ColumnType(dtype="float", unit="")
    HIL_MSL_ERR: ColumnType = ColumnType(dtype="float", unit="")

    def __getitem__(self, name):
        if hasattr(self, name):
            return self.__getattribute__(name)
        else:
            raise KeyError

    @property
    def required_columns(self):
        req = []
        for field in self.__fields__:
            if getattr(self, field).required:
                req.append(field)
        return req

    @property
    def optional_columns(self):
        req = []
        for field in self.__fields__:
            if not getattr(self, field).required:
                req.append(field)
        return req

    def run(self, table):
        for key in self.required_columns:
            self[key].validate_column(table[key])
        for key in self.optional_columns:
            if key in table.colnames:
                self[key].validate_column(table[key])
        return table


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
    HDUCLASS: str
    HDUDOC: str
    HDUVERS: str
    HDUCLAS1: str

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


class GADFEvents(BaseModel):
    header: GADFEventsHeader
    table: Table

    class Config:
        arbitrary_types_allowed = True
        validate_all = True
        validate_assignment = True
        extra = "forbid"

    @validator("table")
    def validate_gadf_events_table(cls, v):
        return GADFTableValidator().run(v)

    @classmethod
    def read(cls, filename, hdu="EVENTS"):
        filename = make_path(filename)
        table = Table.read(filename, hdu)

        hdr = GADFEventsHeader.from_header(table.meta)
        table.meta = None
        return cls(header=hdr, table=table)

    def to_table_hdu(self):
        """Export to table HDU."""
        table_hdu = fits.BinTableHDU(self.table, name="EVENTS")
        table_hdu.header.update(self.header.to_header())
        return table_hdu
