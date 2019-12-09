import sys
sys.path.append(sys.path[0][:-5])

import numpy as np
import autodiff.optimization as opt
from autodiff.autodiff import AutoDiff as ad

def test_conjugate_gradient():
    x = ad(1., [1., 0.,])
    y = ad(1., [0., 1.,])

    fn = lambda x: x[0]**2 + x[1]**2
    output = opt.conjugate_gradient(fn, [x, y], tol=1e-12)
    assert np.linalg.norm(output[0]) < 10e-12

    x = ad(1., [1., 0.,])
    y = ad(1., [0., 1.,])
    fn = lambda x: x[0].cos()**2 + x[1].sin()**2
    output = opt.conjugate_gradient(fn, [x, y], tol=1e-10)
    assert np.linalg.norm(output[0] - np.array([np.pi/2, 0])) < 10e-10


def test_gradient_descent():
    x = ad(1., [1., 0.,])
    y = ad(1., [0., 1.,])

    fn = lambda x: x[0]**2 + x[1]**2
    output = opt.gradient_descent(fn, [x, y], tol=1e-12)
    assert np.linalg.norm(output[0]) < 10e-12

    x = ad(1., [1., 0.,])
    y = ad(1., [0., 1.,])
    fn = lambda x: x[0].cos()**2 + x[1].sin()**2
    output = opt.gradient_descent(fn, [x, y], tol=1e-10)
    assert np.linalg.norm(output[0] - np.array([np.pi/2, 0])) < 10e-10
    

def test_BFGS():
    x = ad(1., [1., 0.,])
    y = ad(1., [0., 1.,])

    fn = lambda x: x[0]**2 + x[1]**2
    output = opt.BFGS(fn, [x, y], tol=1e-12)
    assert np.linalg.norm(output[0]) < 10e-12

    x = ad(1., [1., 0.,])
    y = ad(1., [0., 1.,])
    fn = lambda x: x[0].cos()**2 + x[1].sin()**2
    output = opt.BFGS(fn, [x, y], tol=1e-10)
    assert np.linalg.norm(output[0] - np.array([np.pi/2, 0])) < 10e-10


if __name__ == '__main__':
    test_conjugate_gradient()
    test_gradient_descent()
    test_BFGS()
    
