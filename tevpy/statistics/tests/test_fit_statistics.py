import numpy as np
from numpy.testing import assert_almost_equal
from .. import fit_statistics

def print_info(label, data):
    print('*** %s ***' % label)
    print('Sum: %g' % data.sum())
    print(data)

def test_likelihood_stats():
    # Create some example input
    M = np.array([[0, 0.5, 1, 2],[10, 10.5, 100.5, 1000]])
    print_info('M', M)
    np.random.seed(0)
    D = np.random.poisson(M)
    print_info('D', D)
    
    print_info('cash', fit_statistics.cash(D, M))
    print_info('cstat', fit_statistics.cstat(D, M))

def test_chi2_stats():
    A_S = np.array([[0, 0.5, 1, 2],[10, 10.5, 100.5, 1000]])
    A_B = A_S
    N_S = A_S
    N_B = A_S
    print_info('chi2datavar', fit_statistics.chi2datavar(N_S, N_B, A_S, A_B)) 

if __name__ == '__main__':
    test_likelihood_stats()
    test_chi2_stats()
