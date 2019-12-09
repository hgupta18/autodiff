import sys
sys.path.append(sys.path[0][:-21])

import numpy as np
from autodiff.autodiff import AutoDiff as ad 

def conjugate_gradient(f, num, step_size=0.01, tol=10e-8, max_iter=10000):
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

    # 0th step values for the algorithm
    g = f(num).der
    s = -np.copy(g)

    # Initialize the variables for the loop
    last = np.ones(len(num))*999
    iterations = 0

    while(np.linalg.norm(last - num) > tol):
        last = np.copy(num)

        # Update current value
        num += step_size*s

        # Update gradient
        new_g = f(num).der

        # Udate intermediate values of algorithm
        beta = np.outer(new_g,new_g)/np.dot(g,g)
        s = -g + np.dot(beta, s)
        g = np.copy(new_g)

        iterations += 1

        # Break out of loop if we have reached maximum iterations
        if(iterations > max_iter):
            return num, False, iterations

    return num, True, iterations


if __name__ == '__main__':
    x = ad(1, [1., 0.])
    y = ad(1, [0., 1.])

    fn = lambda x: (x[0]-3)**2 + (x[1] + 1)**2
    print(conjugate_gradient(fn, [x,y]))
