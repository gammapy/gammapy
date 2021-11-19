import copy


class HpxConv:
    """Data structure to define how a HEALPIX map is stored to FITS."""

    def __init__(self, convname, **kwargs):
        self.convname = convname
        self.colstring = kwargs.get("colstring", "CHANNEL")
        self.firstcol = kwargs.get("firstcol", 1)
        self.hduname = kwargs.get("hduname", "SKYMAP")
        self.bands_hdu = kwargs.get("bands_hdu", "EBOUNDS")
        self.quantity_type = kwargs.get("quantity_type", "integral")
        self.frame = kwargs.get("frame", "COORDSYS")

    def colname(self, indx):
        return f"{self.colstring}{indx}"

    @classmethod
    def create(cls, convname="gadf"):
        return copy.deepcopy(HPX_FITS_CONVENTIONS[convname])

    @staticmethod
    def identify_hpx_format(header):
        """Identify the convention used to write this file."""
        # Hopefully the file contains the HPX_CONV keyword specifying
        # the convention used
        if "HPX_CONV" in header:
            return header["HPX_CONV"].lower()

        # Try based on the EXTNAME keyword
        hduname = header.get("EXTNAME", None)
        if hduname == "HPXEXPOSURES":
            return "fgst-bexpcube"
        elif hduname == "SKYMAP2":
            if "COORDTYPE" in header.keys():
                return "galprop"
            else:
                return "galprop2"
        elif hduname == "xtension":
            return "healpy"
        # Check the name of the first column
        colname = header["TTYPE1"]
        if colname == "PIX":
            colname = header["TTYPE2"]

        if colname == "KEY":
            return "fgst-srcmap-sparse"
        elif colname == "ENERGY1":
            return "fgst-template"
        elif colname == "COSBINS":
            return "fgst-ltcube"
        elif colname == "Bin0":
            return "galprop"
        elif colname == "CHANNEL1" or colname == "CHANNEL0":
            if hduname == "SKYMAP":
                return "fgst-ccube"
            else:
                return "fgst-srcmap"
        else:
            raise ValueError("Could not identify HEALPIX convention")


HPX_FITS_CONVENTIONS = {}
"""Various conventions for storing HEALPIX maps in FITS files"""
HPX_FITS_CONVENTIONS[None] = HpxConv("gadf", bands_hdu="BANDS")
HPX_FITS_CONVENTIONS["gadf"] = HpxConv("gadf", bands_hdu="BANDS")
HPX_FITS_CONVENTIONS["fgst-ccube"] = HpxConv("fgst-ccube")
HPX_FITS_CONVENTIONS["fgst-ltcube"] = HpxConv(
    "fgst-ltcube", colstring="COSBINS", hduname="EXPOSURE", bands_hdu="CTHETABOUNDS"
)
HPX_FITS_CONVENTIONS["fgst-bexpcube"] = HpxConv(
    "fgst-bexpcube", colstring="ENERGY", hduname="HPXEXPOSURES", bands_hdu="ENERGIES"
)
HPX_FITS_CONVENTIONS["fgst-srcmap"] = HpxConv(
    "fgst-srcmap", hduname=None, quantity_type="differential"
)
HPX_FITS_CONVENTIONS["fgst-template"] = HpxConv(
    "fgst-template", colstring="ENERGY", bands_hdu="ENERGIES"
)
HPX_FITS_CONVENTIONS["fgst-srcmap-sparse"] = HpxConv(
    "fgst-srcmap-sparse", colstring=None, hduname=None, quantity_type="differential"
)
HPX_FITS_CONVENTIONS["galprop"] = HpxConv(
    "galprop",
    colstring="Bin",
    hduname="SKYMAP2",
    bands_hdu="ENERGIES",
    quantity_type="differential",
    frame="COORDTYPE",
)
HPX_FITS_CONVENTIONS["galprop2"] = HpxConv(
    "galprop",
    colstring="Bin",
    hduname="SKYMAP2",
    bands_hdu="ENERGIES",
    quantity_type="differential",
)
HPX_FITS_CONVENTIONS["healpy"] = HpxConv("healpy", hduname=None, colstring=None)
