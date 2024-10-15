# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import NoOverlapError
from astropy.time import Time
from regions import PointSkyRegion
from gammapy.maps import HpxNDMap, Map, MapAxis, RegionNDMap
from gammapy.maps.hpx.io import HPX_FITS_CONVENTIONS, HpxConv
from gammapy.utils.scripts import make_path
from gammapy.utils.time import time_ref_from_dict, time_ref_to_dict
from . import LightCurveTemplateTemporalModel, Models, SkyModel, TemplateSpatialModel

__all__ = ["read_hermes_cube"]


def _template_model_from_cta_sdc(filename, t_ref=None):
    """To create a `LightCurveTemplateTemporalModel` from the energy-dependent temporal model files of the cta-sdc1.

    This format is subject to change.
    """
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

        if t_ref is None:
            t_ref = Time(55555.5, format="mjd", scale="tt")
        time_header.update(time_ref_to_dict(t_ref, t_ref.scale))
        time_min = time_hdu.data["Initial Time"]
        time_max = time_hdu.data["Final Time"]
        edges = np.append(time_min, time_max[-1]) * u.Unit(time_header["TUNIT1"])
        data = hdul["SPECTRA"]

        time_ref = time_ref_from_dict(time_header, scale=t_ref.scale)
        time_axis = MapAxis.from_edges(edges=edges, name="time", interp="log")

        reg_map = RegionNDMap.create(
            region=PointSkyRegion(center=position),
            axes=[energy_axis, time_axis],
            data=np.array(list(data.data) * u.Unit(data.header["UNITS"])),
        )
    return LightCurveTemplateTemporalModel(reg_map, t_ref=time_ref, filename=filename)


def read_hermes_cube(filename):
    """Read 3d templates produced with hermes."""
    # add hermes conventions to the list used by gammapy
    hermes_conv = HpxConv(
        convname="hermes-template",
        colstring="TFLOAT",
        hduname="xtension",
        frame="COORDTYPE",
        quantity_type="differential",
    )
    HPX_FITS_CONVENTIONS["hermes-template"] = hermes_conv

    maps = []
    energy_nodes = []
    with fits.open(filename) as hdulist:
        # cannot read directly in 3d with Map.read because BANDS HDU is missing
        # https://gamma-astro-data-formats.readthedocs.io/en/v0.2/skymaps/index.html#bands-hdu
        # so we have to loop over hdus and create the energy axis
        for hdu in hdulist[1:]:
            template = HpxNDMap.from_hdu(hdu, format="hermes-template")
            # fix missing/incompatible infos
            template._unit = u.Unit(hdu.header["TUNIT1"])  # .from_hdu expect "BUNIT"
            if template.geom.frame == "G":
                template._geom._frame = "galactic"
            maps.append(template)
            energy_nodes.append(hdu.header["ENERGY"])  # SI unit (see header comment)
    # create energy axis and set unit
    energy_nodes *= u.Joule
    energy_nodes = energy_nodes.to("GeV")
    axis = MapAxis(
        energy_nodes, interp="log", name="energy_true", node_type="center", unit="GeV"
    )
    return Map.from_stack(maps, axis=axis)


def cutout_template_models(models, cutout_kwargs, datasets_names=None):
    """Apply cutout to template models."""
    models_cut = Models()
    if models is None:
        return models_cut
    for m in models:
        if isinstance(m.spatial_model, TemplateSpatialModel):
            try:
                map_ = m.spatial_model.map.cutout(**cutout_kwargs)
            except (NoOverlapError, ValueError):
                continue
            template_cut = TemplateSpatialModel(
                map_,
                normalize=m.spatial_model.normalize,
            )
            model_cut = SkyModel(
                spatial_model=template_cut,
                spectral_model=m.spectral_model,
                datasets_names=datasets_names,
            )
            models_cut.append(model_cut)
        else:
            models_cut.append(m)
    return models_cut
