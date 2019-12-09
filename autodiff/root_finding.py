import sys
from autodiff import AutoDiff as ad
import numpy as np


def _inv_jacobian(num):
    jac = np.array([list(n.der) for n in num])
    return np.linalg.pinv(jac)


def newton(func, num, tol=1e-10, max_iter=10000, return_trace=False):
    '''
        This function runs Newton's method of root finding.

        INPUTS:
          num: np.ndarray of AutoDiff objects
               an autodiff number object

          func: callable, input is array of AutoDiff objects
                the function we are trying to find the root of

          tol: float, optional (default = 1e-10)
               Convergence tolerance

          max_iter: int, optional (default = 1e-10)
                    Maximum number of iterations before no convergence is declared

          return_trace: boolean, optional (default = False)
                        Returns trace of minimization procedure if True

        OUTPUT:
            root: AutoDiff object
                  The value of the root and derivative at the root.

        NOTE:
            This function is currently only implemented for the 1D case.
            When a function has no root, or a root that cannot be reached,
            the function returns "NO ROOT FOUND"

            When the derivative is 0 the functions raises a floating point error with the
            message "ZERO DERIVATIVE"
            When a the starting point is the root, this function returns the root immediately.

            All of the same failure modes apply here as in the mathematical forumlation.
            First, you must start in a good starting position, otherwise the algorithm may not
            converge. Also, the function must approach the root smooth enough to converge.

            Function output must be a list or array, even for scalar valued functions.
            For example:

            >>> lambda x: x[0] # This throws an error because the output is a scalar
            >>> lambda x: [x[0]] # This works because the output is a list
            
    '''
    last = np.array([-999]*len(num))
    root = [n.val for n in num]
    default_der = np.copy([num[i].der for i in range(len(num))])
    if(return_trace):
        trace = [num]

    # Started at root case
    if(isinstance(func(num), ad)):
        err_str = "Function output must be list, even for scalar functions."
        err_str += "\nTry returning your function output as a list: return [output]."
        raise TypeError(err_str)

    f_val = np.copy([f.val for f in func(num)])

    if(np.linalg.norm(f_val) < tol):
        return num

    # Root finding algorithm. Terminates after 200 steps with error message
    iterations = 0
    while(np.linalg.norm(f_val) > 1e-10):

        last = np.copy([n.val for n in num])
        num = func(num)

        # Catch zero derivatives
        if(len(num)==1 and np.linalg.norm([n.der for n in num]) == 0):
            raise FloatingPointError("ZERO DERIVATIVE")

        root -= np.dot(_inv_jacobian(num), num)
        num = np.copy([ad(v.val, default_der[j]) for j, v in enumerate(root)])

        f_val = np.copy([f.val for f in func(num)])

        if(return_trace):
            trace.append(num)
        iterations += 1
        if(iterations == max_iter):
            if(return_trace):
                return num, False, iterations, trace
            else:
                return num, False, iterations

    if(return_trace):
        return num, False, iterations, trace
    else:
        return num, False, iterations


if __name__ == '__main__':
    x = ad(1, [1., 0., 0.])
    y = ad(1, [0., 1., 0.])
    z = ad(1, [0., 0., 1.])
    fn = lambda x: [(x[1]-1)**3, x[2]**3, x[0]**3]
    print(newton(fn, [x,y,z]))

