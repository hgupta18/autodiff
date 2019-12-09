import sys
sys.path.append(sys.path[0][:-21])

import numpy as np
from autodiff.autodiff import AutoDiff as ad


def broyden(func, num, init_jacobian=1, tol=1e-10, max_iter=10000):
    '''
        INPUTS:
            func:
            num:
            init_jacobian:
            tol:
            max_iter:

        OUTPUTS:
            root:
            convergence:
            iterations:

        NOTES:
            pass
    '''
    if(init_jacobian == 1):
        init_jacobian = np.zeros((len(num), len(num[0].der)))
        smaller = min(init_jacobian.shape)
        init_jacobian[:smaller, :smaller] += np.eye(smaller)
        print(init_jacobian)
    


if __name__ == '__main__':
    x = ad(1, [1., 0., 0.,])
    y = ad(1, [0., 1., 0.,])
    z = ad(1, [0., 0., 1.,])

    fn = lambda x: [x[0]+2*x[1] - 1, x[2]]

    print(broyden(fn, [x,y,z]))

