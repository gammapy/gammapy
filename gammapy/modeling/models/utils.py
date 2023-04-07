import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from regions import PointSkyRegion
from gammapy.maps import MapAxis, RegionNDMap
from gammapy.utils.scripts import make_path
from gammapy.utils.time import time_ref_from_dict
from . import LightCurveTemplateTemporalModel


def _template_model_from_cta_sdc(filename):
    """To create a `LightCurveTemplateTemporalModel`
    from the cta-sdc files. This format is subject to change"""
    filename = str(make_path(filename))
    with fits.open(filename) as hdul:
        frame = hdul[0].header.get("frame", "icrs")
        position = SkyCoord(
            hdul[0].header["LONG"] * u.deg, hdul[0].header["LAT"] * u.deg, frame=frame
        )

        energy_hdu = hdul["ENERGIES"]
        energy_axis = MapAxis.from_nodes(
            nodes=energy_hdu.data,
            unit=energy_hdu.header["TUNIT1"],
            name="energy",
            interp="log",
        )
        time_hdu = hdul["TIMES"]
        time_header = time_hdu.header
        time_header.setdefault("MJDREFF", 0.5)
        time_header.setdefault("MJDREFI", 55555)
        time_header.setdefault("scale", "tt")
        time_min = time_hdu.data["Initial Time"]
        time_max = time_hdu.data["Final Time"]
        edges = np.append(time_min, time_max[-1]) * u.Unit(time_header["TUNIT1"])
        data = hdul["SPECTRA"]

        time_ref = time_ref_from_dict(time_header)
        time_axis = MapAxis.from_edges(edges=edges, name="time", interp="log")

        reg_map = RegionNDMap.create(
            region=PointSkyRegion(center=position),
            axes=[energy_axis, time_axis],
            data=np.array(list(data.data) * u.Unit(data.header["UNITS"])),
        )
    return LightCurveTemplateTemporalModel(reg_map, t_ref=time_ref, filename=filename)
