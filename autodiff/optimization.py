import sys
sys.path.append(sys.path[0][:-21])

import numpy as np
from autodiff import autodiff as ad

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
    '''

    # If hessian guess is default value, set to identity
    print("WHY")
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
        y = func(num).der + df_val
        init_hessian += _hessian_update(init_hessian, y, s)
        iterations += 1
        if(iterations == max_iter):
            return num, False, iterations
    
    return num, True, iterations


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

    for i in range(max_iter):

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
    x = ad(1, [1., 0.])
    y = ad(1, [0., 1.])
    fn = lambda x: (x[0]-2)**2 + (x[1]+1)**2
    _ = BFGS(fn, [x, y])
    print(_)

    x = ad(1, [1., 0.])
    y = ad(1, [0., 1.])

    fn = lambda x: (x[0]-3)**2 + (x[1] + 1)**2
    print(conjugate_gradient(fn, [x,y]))

    xs = [1]
    ys = [1]
    x = ad(1, [1.,0.,0.])
    y = ad(1, [0.,1.,0.])
    z = ad(1, [0.,0.,1.])

    fn = lambda x: (x[0].cos()+1)**2 + x[1]**2 + (x[2]-2)**2
    print(gradient_descent(fn, [x, y, z], step_size=0.8))

