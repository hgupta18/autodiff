import sys
sys.path.append(sys.path[0][:-21])

import numpy as np
from autodiff.autodiff import AutoDiff as ad


def gradient_descent(f, x, steps=100000, step_size=0.1, tol=10e-8):
    '''
        INPUTS:
          f - the function we would like to perform gradient descent on
          x - vector of inputs as an AutoDiff object
          step - step size of algorithm
          tol - convergence critera. if 

        OUTPUTS:
          min - the minimum to which gradient descent converged
          converged - Boolean indicating whether or not the method converged

        NOTE:
          Need to catch failure modes
    '''

    n_func = len(x)
    default_der = np.copy([x[i].der for i in range(n_func)])
    last = np.copy(f(x).val)
    for i in range(steps):

        # Evaluate function to get derivatives
        f_val = f(x)

        # Update values by stepping along derivative
        for j in range(n_func):
            x[j] -= ad(step_size*f_val.der[j])

        # Recast values as autodiff objects
        for j in range(n_func):
            x[j] = ad(x[j].val, default_der[j])

        # Check if converged
        if(np.linalg.norm(x - last) < tol):
            return [x, True, i]

        last = np.copy(x)

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

