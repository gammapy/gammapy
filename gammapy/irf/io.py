# Licensed under a 3-clause BSD style license - see LICENSE.rst
from .background import Background3D
from .effective_area import EffectiveAreaTable2D
from .energy_dispersion import EnergyDispersion2D
from .psf_gauss import EnergyDependentMultiGaussPSF

__all__ = ["load_cta_irfs"]


IRF_DL3_AXES_SPECIFICATION = {
    "THETA": {"name": "offset", "interp": "lin"},
    "ENERG": {"name": "energy_true", "interp": "log"},
    "ETRUE": {"name": "energy_true", "interp": "log"},
    "RAD": {"name": "rad", "interp": "lin"},
    "DETX": {"name": "fov_lon", "interp": "lin"},
    "DETY": {"name": "fov_lat", "interp": "lin"},
    "MIGRA": {"name": "migra", "interp": "lin"},
}


def load_cta_irfs(filename):
    """load CTA instrument response function and return a dictionary container.

    The IRF format should be compliant with the one discussed
    at http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/.

    The various IRFs are accessible with the following keys:

    - 'aeff' is a `~gammapy.irf.EffectiveAreaTable2D`
    - 'edisp'  is a `~gammapy.irf.EnergyDispersion2D`
    - 'psf' is a `~gammapy.irf.EnergyDependentMultiGaussPSF`
    - 'bkg' is a  `~gammapy.irf.Background3D`

    Parameters
    ----------
    filename : str
        the input filename. Default is

    Returns
    -------
    cta_irf : dict
        the IRF dictionary

    Examples
    --------
    Access the CTA 1DC IRFs stored in the gammapy datasets

    .. code-block:: python

        from gammapy.irf import load_cta_irfs
        cta_irf = load_cta_irfs("$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits")
        print(cta_irf['aeff'])
    """
    aeff = EffectiveAreaTable2D.read(filename, hdu="EFFECTIVE AREA")
    bkg = Background3D.read(filename, hdu="BACKGROUND")
    edisp = EnergyDispersion2D.read(filename, hdu="ENERGY DISPERSION")
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu="POINT SPREAD FUNCTION")

    return dict(aeff=aeff, bkg=bkg, edisp=edisp, psf=psf)
