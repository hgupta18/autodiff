import sys
sys.path.append(sys.path[0][:-21])

import numpy as np
from autodiff.autodiff import AutoDiff as ad


def gradient_descent(func, num, step_size=0.1, tol=10e-8, max_iter=10000):
    '''
        INPUTS:
          func: callable (function)
                 the function we would like to perform gradient descent on

          num: np.ndarray of AutoDiff objects
                vector of inputs a list of AutoDiff objects

          step_size: float, optional (default = 0.1)
                      step size of algorithm

          tol: float, optional (default = 1e-10)
                convergence critera. if 

          max_iter: int, optional (default = 10000)
                    number of iterations allowed before no convergence is declared


        OUTPUTS:
          min: np.ndarray of AutoDiff objects
               the minimum to which gradient descent converged

          converged: boolean
                     True if the algorithm comverge, else False
          iterations: int
                      The number of iterations performed, whether or not the algorithm converged

        NOTE:
          This algorithm terminates when the gradient is zero, whether or not that point
          is a minimum or maximum. The algorithm is fairly robust and should converge
          eventually, depending on step size. Default parameters are set so that most
          functions used for input should converge.
    '''

    # Set up values
    n_func = len(num)  # Number of functions in the vector
    default_der = np.copy([num[i].der for i in range(n_func)]) # Used for casting back to AD
    last = np.copy(func(num).val)  # Last value, used to check convergence

    for i in range(steps):

        # Evaluate function to get derivatives
        f_val = func(num)

        # Update values by stepping along derivative
        for j in range(n_func):
            num[j] -= ad(step_size*f_val.der[j])

        # Recast values as autodiff objects
        for j in range(n_func):
            num[j] = ad(num[j].val, default_der[j])

        # Check if converged
        if(np.linalg.norm(num - last) < tol):
            return [num, True, i]

        last = np.copy(num)

    return [x, False, i]


if __name__ == '__main__':
    xs = [1]
    ys = [1]
    x = ad(1, [1.,0.,0.])
    y = ad(1, [0.,1.,0.])
    z = ad(1, [0.,0.,1.])

    fn = lambda x: (x[0].cos()+1)**2 + x[1]**2 + (x[2]-2)**2
    print(gradient_descent(fn, [x, y, z], step_size=0.8))
    #f_val = f(x, y)

