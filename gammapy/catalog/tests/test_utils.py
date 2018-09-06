# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.coordinates import SkyCoord, Longitude, Latitude
from ...catalog import coordinate_iau_format, ra_iau_format, dec_iau_format


def test_make_source_designation():
    # Crab pulsar position for HESS
    coordinate = SkyCoord("05h34m31.93830s +22d00m52.1758s", frame="icrs")
    strrep = coordinate_iau_format(coordinate, ra_digits=4)
    assert strrep == "0534+220"

    # PKS 2155-304 AGN position for 2FGL
    coordinate = SkyCoord("21h58m52.06511s -30d13m32.1182s", frame="icrs")
    strrep = coordinate_iau_format(coordinate, ra_digits=5)
    assert strrep == "2158.8-3013"

    # Check the example from Section 3.2.1 of the IAU spec:
    # http://cdsweb.u-strasbg.fr/Dic/iau-spec.html
    icrs = SkyCoord("00h51m09.38s -42d26m33.8s", frame="icrs")
    fk4 = icrs.transform_to("fk4")

    strrep = coordinate_iau_format(icrs, ra_digits=6)
    assert strrep == "005109-4226.5"

    strrep = coordinate_iau_format(fk4, ra_digits=6)
    assert strrep == "004848-4242.8"

    strrep = coordinate_iau_format(fk4, ra_digits=4)
    assert strrep == "0048-427"

    strrep = coordinate_iau_format(fk4, ra_digits=4, dec_digits=2)
    assert strrep == "0048-42"

    # Check that array coordinate input works
    coordinates = SkyCoord(
        ra=[10.68458, 83.82208], dec=[41.26917, -5.39111], unit=("deg", "deg")
    )
    strreps = coordinate_iau_format(coordinates, ra_digits=5, prefix="HESS J")
    assert strreps == ["HESS J0042.7+4116", "HESS J0535.2-0523"]


def test_ra_iau_format():
    # Test various number of digits (output not verified)
    ra = Longitude("05h34m31.93830s")
    assert ra_iau_format(ra, digits=2) == "05"
    assert ra_iau_format(ra, digits=3) == "055"
    assert ra_iau_format(ra, digits=4) == "0534"
    assert ra_iau_format(ra, digits=5) == "0534.5"
    assert ra_iau_format(ra, digits=6) == "053431"
    assert ra_iau_format(ra, digits=7) == "053431.9"
    assert ra_iau_format(ra, digits=8) == "053431.93"

    ra = Longitude("00h51m09.38s")
    assert ra_iau_format(ra, digits=2) == "00"
    assert ra_iau_format(ra, digits=3) == "008"
    assert ra_iau_format(ra, digits=4) == "0051"
    assert ra_iau_format(ra, digits=5) == "0051.1"
    assert ra_iau_format(ra, digits=6) == "005109"
    assert ra_iau_format(ra, digits=7) == "005109.3"
    # This is subject to rounding errors ... so we skip it:
    # hms_tuple(h=0.0, m=51.0, s=9.3799999999997397)
    # assert ra_iau_format(ra, digits=8) == '005109.38'


def test_dec_iau_format():
    # Test various number of digits (output not verified)
    dec = Latitude("+22d00m52.1758s")
    assert dec_iau_format(dec, digits=2) == "+22"
    assert dec_iau_format(dec, digits=3) == "+220"
    assert dec_iau_format(dec, digits=4) == "+2200"
    assert dec_iau_format(dec, digits=5) == "+2200.8"
    assert dec_iau_format(dec, digits=6) == "+2200.52"
    assert dec_iau_format(dec, digits=7) == "+220052.1"
    assert dec_iau_format(dec, digits=8) == "+220052.17"

    dec = Latitude("-42d26m33.8s")
    assert dec_iau_format(dec, digits=2) == "-42"
    assert dec_iau_format(dec, digits=3) == "-424"
    assert dec_iau_format(dec, digits=4) == "-4226"
    assert dec_iau_format(dec, digits=5) == "-4226.5"
    assert dec_iau_format(dec, digits=6) == "-4226.33"
    assert dec_iau_format(dec, digits=7) == "-422633.7"
    assert dec_iau_format(dec, digits=8) == "-422633.79"
