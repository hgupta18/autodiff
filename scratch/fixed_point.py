import sys
sys.path.append(sys.path[0][:-21])

import pytest
from autodiff.autodiff import AutoDiff as ad
import numpy as np


def fixed_point(func, num, tol=1e-10, max_iter=10000):
    '''
        This function runs fixed point iteration for root finding.

        INPUTS:
          num: an autodiff number object
          func: the function we are trying to find the root of
          tol: Convergence tolerance

        OUTPUT:
            The value of the root.

        NOTE:
            This function is currently only implemented for the 1D case.
            When a function has no root, or a root that cannot be reached,
            the function returns "NO ROOT FOUND"

            When the derivative is 0 the functions raises a floating point error with the
            message "ZERO DERIVATIVE"
            When a the starting point is the root, this function returns the root immediately.
    '''
    #last = np.array([-999]*len(num))
    root = [n.val for n in num]
    default_der = np.copy([num[i].der for i in range(len(num))])

    # Started at root case
    n_val = np.copy([n.val for n in num])
    f_val = np.copy([f.val for f in func(num)])
    if(np.linalg.norm(f_val) < tol):
        return num

    # Comparison used to determine convergence
    last = np.ones(len(num))*999

    # Root finding algorithm. Terminates after 200 steps with error message
    iterations = 0
    while(np.linalg.norm(last - num) > 1e-30):

        last = np.copy(num)
        num = func(num)
        print(num)
        num = np.copy([ad(n.val, default_der[j]) for j, n in enumerate(num)])

        iterations += 1
        if(iterations == max_iter):
            return num, False, iterations

    return num, True, iterations


if __name__ == '__main__':
    x = ad(0.3, [1., 0., 0.])
    y = ad(0.9, [0., 1., 0.])
    #z = ad(0.9, [0., 0., 1.])
    fn = lambda x: [(x[0]-0.2)**2]#, (x[1])**2]#, (x[2]+0.1)**2]
    print(fixed_point(fn, [x]))

