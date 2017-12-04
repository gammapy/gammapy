def test_1():
    from gammapy.time.tests.test_lightcurve import lc

    lc = lc()

    print(lc)
    print(lc.table)
    print(lc.table.info())
    import matplotlib.pyplot as plt
    lc.plot()
    plt.show()


def test_2():
    from astropy.table import Table
    from gammapy.time import LightCurve
    url = 'https://github.com/gammapy/gamma-cat/raw/master/input/data/2006/2006A%2526A...460..743A/tev-000119-lc.ecsv'
    table = Table.read(url, format='ascii.ecsv')
    lc = LightCurve(table)
    print(lc.time_min[:2].iso)

    import matplotlib.pyplot as plt
    lc.plot()
    plt.show()


def test_3():
    from gammapy.catalog import SourceCatalog3FGL
    source = SourceCatalog3FGL()['3FGL J0349.9-2102']
    lc = source.lightcurve
    lc.table.info()
    import matplotlib.pyplot as plt
    lc.plot()
    plt.show()


if __name__ == '__main__':
    test_3()
