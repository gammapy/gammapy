# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy.stats as stats
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ..inverse_cdf import InverseCDFSampler, MapEventSampler
from ....cube import MapEvaluator
from ....cube.models import SkyModel
from ....image.models import SkyGaussian
from ....maps import Map, MapAxis
from ....spectrum.models import PowerLaw
from ....time.models import LightCurveTableModel


def uniform_dist(x, a, b):
    return np.select([x <= a, x >= b], [0, 0], 1 / (b - a))


def gauss_dist(x, mu, sigma):
    return stats.norm.pdf(x, mu, sigma)


def po(x):
    return x ** (-1.0 * 2)


def rate(x, c="1e4 s"):
    c = u.Quantity(c)
    return np.exp(-x / c)


def get_npred_map():
    position = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
    energy_axis = MapAxis.from_bounds(
        1, 100, nbin=30, unit="TeV", name="energy", interp="log"
    )

    exposure = Map.create(
        binsz=0.02,
        map_type="wcs",
        skydir=position,
        width="5 deg",
        axes=[energy_axis],
        coordsys="GAL",
        unit="cm2 s",
    )

    spatial_model = SkyGaussian("0 deg", "0 deg", sigma="0.2 deg")
    spectral_model = PowerLaw(amplitude="1e-11 cm-2 s-1 TeV-1")
    skymodel = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    exposure.data = 1e14 * np.ones(exposure.data.shape)
    evaluator = MapEvaluator(model=skymodel, exposure=exposure)

    npred = evaluator.compute_npred()
    return npred


def test_uniform_dist_sampling():
    n_sampled = 1000
    x = np.linspace(-2, 2, n_sampled)

    a, b = -1, 1
    pdf = uniform_dist(x, a=a, b=b)
    sampler = InverseCDFSampler(pdf=pdf, random_state=0)

    idx = sampler.sample(int(1e4))
    x_sampled = np.interp(idx, np.arange(n_sampled), x)

    assert_allclose(np.mean(x_sampled), 0.5 * (a + b), atol=0.01)
    assert_allclose(
        np.std(x_sampled), np.sqrt(1 / 3 * (a ** 2 + a * b + b ** 2)), rtol=0.01
    )


def test_norm_dist_sampling():
    n_sampled = 1000
    x = np.linspace(-2, 2, n_sampled)

    mu, sigma = 0, 0.1
    pdf = gauss_dist(x=x, mu=mu, sigma=sigma)
    sampler = InverseCDFSampler(pdf=pdf, random_state=0)

    idx = sampler.sample(int(1e5))
    x_sampled = np.interp(idx, np.arange(n_sampled), x)

    assert_allclose(np.mean(x_sampled), mu, atol=0.01)
    assert_allclose(np.std(x_sampled), sigma, atol=0.005)


def test_axis_sampling():
    n_sampled = 1000
    x = np.linspace(-2, 2, n_sampled)

    a, b = -1, 1
    pdf_uniform = uniform_dist(x, a=a, b=b)

    mu, sigma = 0, 0.1
    pdf_gauss = gauss_dist(x=x, mu=mu, sigma=sigma)

    pdf = np.vstack([pdf_gauss, pdf_uniform])
    sampler = InverseCDFSampler(pdf, random_state=0, axis=1)

    idx = sampler.sample_axis()
    x_sampled = np.interp(idx, np.arange(n_sampled), x)

    assert_allclose(x_sampled, [0.01042147, 0.43061014], rtol=1e-5)


def test_map_sampling():
    npred = get_npred_map()
    time = np.arange(0, 10, 0.06) * u.hour

    table = Table()
    table["TIME"] = time
    table["NORM"] = rate(time)
    temporal_model = LightCurveTableModel(table)

    t_min = Time("2010-01-01T00:00:00")
    t_max = Time("2010-01-01T08:00:00")

    sampler = MapEventSampler(
        npred,
        t_min=t_min,
        t_max=t_max,
        temporal_model=temporal_model,
        random_state=0,
        t_delta="10 min",
    )
    events = sampler.sample_events(n_events=2)

    assert len(events) == 2
    assert_allclose(events["TIME"].data, [175.035023, 4217.336952], rtol=1e-5)
    assert_allclose(events["RA_TRUE"].data, [266.638497, 266.578664], rtol=1e-5)
    assert_allclose(events["DEC_TRUE"].data, [-28.930393, -28.815534], rtol=1e-5)
    assert_allclose(events["ENERGY_TRUE"].data, [2.755397, 1.72316], rtol=1e-5)

    assert_allclose(events.meta["ONTIME"], 8 * 3600)
