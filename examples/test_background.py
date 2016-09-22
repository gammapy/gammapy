import numpy as np
from astropy.units import Quantity
from gammapy.data import DataStore
from gammapy.background import FOVCube


def test_fill_cube():
    filename = '$GAMMAPY_EXTRA/test_datasets/background/bg_cube_model_test1.fits'
    array = FOVCube.read(filename, format='table', scheme='bg_cube')
    array.data = Quantity(np.zeros_like(array.data.value), 'u')
    print(type(array.data))

    dir = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2'
    data_store = DataStore.from_dir(dir)
    ev_list = data_store.load_all('events')

    array.fill_events(ev_list)

    array.write('test_background.fits', format='image', clobber=True)


if __name__ == '__main__':
    test_fill_cube()
