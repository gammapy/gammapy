"""Test the speed and precision of the sum of many values.

When fitting data with many events or bins (say 1e6 or 1e9 or more), the fit statistic is first
computed per event or bin and then summed.

The sum (and thus the fit statistic and fit results) can become incorrect due to rounding errors
(see e.g. http://en.wikipedia.org/wiki/Kahan_summation_algorithm)

There are clever and slower methods to avoid this problem (like the Kahan summation algorithm),
but they are not readily available in stdlib C / C++ or numpy
(there's an open feature request for numpy though: https://github.com/numpy/numpy/issues/2448)

The simplest solution is to use 128 bit precision for the accumulator in numpy:
In [7]: np.sum([1e10, -1e10, 1e-6] * int(1e6))
Out[7]: 1.9073477254638671
In [8]: np.sum([1e10, -1e10, 1e-6] * int(1e6), dtype='float128')
Out[8]: 1.0002404448965787888

Sherpa uses Kahan summation:
$ASCDS_INSTALL/src/pkg/sherpa/sherpa/sherpa/include/sherpa/stats.hh

This is a quick benchmarking of the precision, speed of sum as a function of these parameters:
* Accumulator bit size: 32, 64 or 128
* Number of elements: 1e6, 1e9
* TODO: how to choose values in a meaningful way? Precision results will completely depend on this.
  Should be chosen similar to typical / extreme fitting cases with CASH, CSTAT, CHI2 fits
  For now we only check the speed.
* TODO: Check against C and Cython implementation
"""

from timeit import Timer

dtypes = ['float32', 'float64', 'float128']
sizes = [int(1e6), int(1e9)]


def setup(size, dtype):
    return """
import numpy as np
data = np.zeros({size}, dtype='{dtype}')
"""[1:-1].format(**locals())


def statement(dtype):
    return """data.sum(dtype='{dtype}')""".format(**locals())

for data_dtype in dtypes:
    for accumulator_dtype in dtypes:
        for size in sizes:
            timer = Timer(statement(accumulator_dtype), setup(size, data_dtype))
            time = min(timer.repeat(repeat=3, number=1))
            # Let's use the frequency in GHz of summed elements as our measure of speed
            speed = 1e-9 * (size / time)
            print('%10s %10s %10d %10.5f' %
                  (data_dtype, accumulator_dtype, size, speed))

"""
On my 2.6 GHz Intel Core I7 Macbook the speed doesn't depend on data or accumulator dtype at all.
This is weird, because it's a 64 bit machine, so 128 bit addition should be slower.
Also for such a simple computation as sum the limiting factor should be memory loading speed,
so 128 bit data should be slower to process than 64 bit data?

In [53]: run sum_benchmark.py
       f32        f32    1000000    0.82793
       f32        f32 1000000000    1.12276
       f32        f64    1000000    1.12207
       f32        f64 1000000000    1.10964
       f32       f128    1000000    1.04155
       f32       f128 1000000000    1.12900
       f64        f32    1000000    1.10609
       f64        f32 1000000000    1.12823
       f64        f64    1000000    1.10493
       f64        f64 1000000000    1.11920
       f64       f128    1000000    1.15450
       f64       f128 1000000000    1.11794
      f128        f32    1000000    1.12087
      f128        f32 1000000000    1.12223
      f128        f64    1000000    1.09885
      f128        f64 1000000000    1.11911
      f128       f128    1000000    1.06943
      f128       f128 1000000000    1.12578
"""
