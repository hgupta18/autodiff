import autodiff.autodiff as ad
import numpy as np

def newton_demo():
    '''
      This function demos a 1D Newton root finding algorithm. In this example we are
      finding the root of f(x) = x^2 with the initial guess x_0 = 2.
    '''
    function = lambda x: x**2    # Function we want root of
    my_number = ad.AutoDiff(2)   # Initial value
    root = my_number.val         # Real component of initial value

    last = -999                  # Initialize loop with obviously incorrect answer
    while(np.abs(my_number.val - last) > 1e-10): # Exit loop once root meets threshold
        last = np.copy(my_number.val)            # Copy current value as last value
        my_number = function(my_number)          # Apply function to current value

        root -= my_number.val/my_number.der      # Newton's method step
        my_number = ad.AutoDiff(root)            # Recast updated root as AutoDiff object

    return my_number.val


if __name__ == '__main__':
    print(newton_demo())
