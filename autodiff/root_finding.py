import sys
sys.path.append(sys.path[0][:-21])

import pytest
from autodiff.autodiff import AutoDiff as ad
import numpy as np


def inv_jacobian(num):
    jac = np.array([list(n.der) for n in num])
    return np.linalg.pinv(jac)


def newton(func, num, tol=1e-10):
    '''
        This function runs Newton's method of root finding.

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
    last = np.array([-999]*len(num))
    root = [n.val for n in num]
    default_der = np.copy([num[i].der for i in range(len(num))])

    # Started at root case
    n_val = np.copy([n.val for n in num])
    f_val = np.copy([f.val for f in func(num)])
    if(np.linalg.norm(f_val) < tol):
        return num

    # Root finding algorithm. Terminates after 200 steps with error message
    iterations = 0
    while(np.linalg.norm(f_val) > 1e-10):

        last = np.copy([n.val for n in num])
        num = func(num)

        # Catch zero derivatives
        if(np.linalg.norm([n.der for n in num]) == 0):
            raise FloatingPointError("ZERO DERIVATIVE")

        root -= np.dot(inv_jacobian(num), num)
        num = np.copy([ad(v.val, default_der[j]) for j, v in enumerate(root)])

        n_val = np.copy([n.val for n in num])
        f_val = np.copy([f.val for f in func(num)])

        iterations += 1
        if(iterations == 2000):
            return num, False, iterations

    return num, True, iterations


if __name__ == '__main__':
    x = ad(1, [1., 0., 0.])
    y = ad(1, [0., 1., 0.])
    z = ad(1, [0., 0., 1.])
    fn = lambda x: [(x[1]-1)**3, x[2]**3, x[0]**3]
    print(newton(fn, [x,y,z]))

