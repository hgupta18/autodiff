import sys
sys.path.append(sys.path[0][:-5])

import numpy as np
import autodiff.autodiff as ad
import autodiff.optimization as opt

def test_conjugate_gradient():
    assert True


def test_gradient_descent():
    assert True


def test_BFGS():
    assert True


if __name__ == '__main__':
    test_conjugate_gradient()
    test_gradient_descent()
    test_BFGS()
    
