# Licensed under a 3-clause BSD style license - see LICENSE.rst
import unittest
"""
from astropy.utils.compat.odict import OrderedDict
from spec.fluxpoints import FluxPoints

class TestFluxPoints(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def testFromDict(self):
        data = odict(x=[1,2], y=[3,4])
        points = FluxPoints.from_dict(data)
        self.assertEqual(points.y[0], 3)
    def testFromAscii(self):
        points = FluxPoints.from_ascii('input/crab_hess_spec.txt')
        # Check row # 13 (when starting counting at 0)
        # against values from ascii file
        # 7.145e+00 1.945e-13 2.724e-14 2.512e-14
        self.assertEqual(points.x[13], 7.145e+00)
        self.assertEqual(points.y[13], 1.945e-13)
        self.assertEqual(points.eyl[13], 2.724e-14)
        self.assertEqual(points.eyh[13], 2.512e-14)
        #print points.row(13)
"""
