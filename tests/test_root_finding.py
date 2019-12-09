import sys
sys.path.append(sys.path[0][:-5])

import numpy as np
import autodiff.root_finding as rf
from autodiff.autodiff import AutoDiff as ad


def test_newton():
    x = ad(1, [1., 0.])
    y = ad(1, [0., 1.])

    fn = lambda x: [x[0], x[1]]

    output = rf.newton(fn, [x,y], tol=1e-10)
    o_vals = [o.val for o in output[0]]
    assert np.linalg.norm(o_vals) < 10e-10


    x = ad(2, [1., 0.])
    y = ad(3, [0., 1.])
    fn = lambda x: [x[0]**2 + x[1]**2]
    output = rf.newton(fn, [x,y], tol=1e-10)
    o_vals = [o.val for o in output[0]]
    assert np.linalg.norm(o_vals) < 10e-6

    x = ad(0.8, [1.])
    fn = lambda x: [x[0].cos()]
    output = rf.newton(fn, [x], tol=1e-10)
    assert np.isclose(output[0][0].val, np.pi/2)


if __name__ == '__main__':
    test_newton()
