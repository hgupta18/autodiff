import sys
sys.path.append(sys.path[0][:-21])

import numpy as np
from autodiff.autodiff import AutoDiff as ad

def _hessian_update(B, y, s):
    '''
        Internal function used in the BFGS algorithm
    '''
    return np.outer(y, y)/np.dot(y, s) - \
           np.dot(B, np.outer(s, np.dot(s.T, B)))/np.dot(s.T, np.dot(B, s))


def BFGS(func, num, init_hessian=None, step_size=0.1, tol=1e-10, max_iter=10000):
    '''
        INPUTS:
          func: callable (function)
                 the function we would like to perform gradient descent on

          num: np.ndarray of AutoDiff objects
                vector of inputs a list of AutoDiff objects

          init_hessian: np.ndarray, optional (default = None)
                        initial guess of the hessian. If None, we set the guess to the
                        identity matrix of size nxn where n is length of num

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

    # If hessian guess is default value, set to identity
    if(init_hessian == None):
        init_hessian = np.eye(len(num))

    # Initialize variables for loop
    last = np.ones(len(num))*999
    iterations = 0

    while(np.linalg.norm(last - num) > tol):
        last = np.copy([v.val for v in num])
        df_val = -func(num).der
        s = np.linalg.solve(init_hessian, df_val)
        num += s
        y = f(num).der + df_val
        init_hessian += _hessian_update(init_hessian, y, s)
        iterations += 1
        if(iterations == max_iter):
            return num, False, iterations
    
    return num, True, iterations


if __name__ == '__main__':
    x = ad(1, [1., 0.])
    y = ad(1, [0., 1.])
    fn = lambda x: (x[0]-2)**2 + (x[1]+1)**2
    _ = BFGS(fn, [x, y])
    print(_)

