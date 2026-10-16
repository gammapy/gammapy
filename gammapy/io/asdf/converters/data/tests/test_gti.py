# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import astropy.units as u
from astropy.table import Table
from astropy.time import Time

from gammapy.data import GTI
from gammapy.utils.testing import assert_time_allclose

asdf = pytest.importorskip("asdf")
pytest.importorskip("asdf.testing")
from asdf.testing.helpers import yaml_to_asdf  # noqa: E402

time_ref = Time("2010-01-01", scale="tt")


def test_gti_roundtrip(tmp_path):
    file_path = tmp_path / "test.asdf"
    times = {
        "START": time_ref + [1, 5, 10, 15] * u.s,
        "STOP": time_ref + [3, 7, 14, 20] * u.s,
    }
    gti = GTI(Table(times), reference_time=time_ref)

    with asdf.AsdfFile() as af:
        af["gti"] = gti
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        result = af["gti"]
        assert isinstance(result.time_start, Time)
        assert isinstance(result.time_stop, Time)
        assert_time_allclose(result.time_start, gti.time_start)
        assert_time_allclose(result.time_stop, gti.time_stop)
        assert_time_allclose(result.time_ref, gti.time_ref)


tested_read_gti_example = [
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/data/gti-1.0.0>
         reference_time: !time/time-1.4.0 {scale: tt, value: '2010-01-01 00:00:00.000'}
          """,
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/data/gti-1.0.0>
        table: "not a table"
        reference_time: !time/time-1.4.0 {scale: tt, value: '2010-01-01 00:00:00.000'}
        """,
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/data/gti-1.0.0>
        table: !<tag:astropy.org:astropy/table/table-1.3.0> {}
        reference_time: "2010-01-01"
        """,
    },
]


@pytest.mark.parametrize("example", tested_read_gti_example)
def test_gti_read_examples(example):
    buff = yaml_to_asdf(f"example: {example['example'].strip()}")
    with pytest.raises(asdf.exceptions.ValidationError):
        asdf.open(buff)
