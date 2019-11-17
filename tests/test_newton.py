import sys
sys.path.append(sys.path[0][:-5])

import pytest
import makeADifference.dual as dual
import autodiff.autodiff as ad
import numpy as np


def newton(num, func):
    '''
        This function runs Newton's method of root finding.

        INPUTS:
          num: an autodiff number object
          func: the function we are trying to find the root of

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
    last = -999
    root = num.val

    # Started at root case
    if(np.abs(func(num).val) < 1e-10):
        return num.val

    # Root finding algorithm. Terminates after 200 steps with error message
    iterations = 0
    while(np.abs(num.val-last) > 1e-10):
        last = np.copy(num.val)
        num = func(num)

        # Catch zero derivatives
        if(np.abs(num.der) == 0):
            print("MY NUM: {}".format(num))
            raise FloatingPointError("ZERO DERIVATIVE")

        root -= num.val/num.der
        num = ad.AutoDiff(root)
        iterations += 1
        if(iterations == 200):
            return "NO ROOT FOUND"
    return num.val


def test_pow():
    '''
        Testing the power function for multiple inputs and functions.
        The functions tested are:
            x^2, x^3, (x-1)^2, (x+1)^2, (x-1)^3, (x+1)^3, x^2-1, x^3-1, x^2+1, x^3+1

        Each function is tested to the left and right of each root.

        The function x^2 + 1 is a failure mode because it has no real root.
        I have not decided what to do with this yet.
    '''
    def power(x, exp, lr_shift=0, ud_shift=0):
        return (x-lr_shift)**exp + ud_shift

    # Starting to the right of the root
    start = 2
    my_num = ad.AutoDiff(start)
    assert np.isclose(newton(my_num, lambda x: power(x, 2)), 0) # x^2
    assert np.isclose(newton(my_num, lambda x: power(x, 3)), 0) # x^3
    assert np.isclose(newton(my_num, lambda x: power(x, 2, 1)), 1) # (x-1)^2
    assert np.isclose(newton(my_num, lambda x: power(x, 3, 1)), 1) # (x-1)^3
    assert np.isclose(newton(my_num, lambda x: power(x, 2, -1)), -1) # (x+1)^2
    assert np.isclose(newton(my_num, lambda x: power(x, 3, -1)), -1) # (x+1)^3

    assert newton(my_num, lambda x: power(x, 2, 0, 1)) == "NO ROOT FOUND" # x^2 + 1
    assert np.isclose(newton(my_num, lambda x: power(x, 3, 0, 1)), -1) # x^3 + 1
    assert np.isclose(newton(my_num, lambda x: power(x, 2, 0, -1)), 1) # x^2 - 1, root=1
    assert np.isclose(newton(my_num, lambda x: power(x, 3, 0, -1)), 1) # x^3 - 1

    my_num = ad.AutoDiff(-0.5)
    assert np.isclose(newton(my_num, lambda x: power(x, 2, 0, -1)), -1) # x^2 - 1, root=-1

    # Starting to the right of the root
    start = -2
    my_num = ad.AutoDiff(start)
    assert np.isclose(newton(my_num, lambda x: power(x, 2)), 0) # x^2
    assert np.isclose(newton(my_num, lambda x: power(x, 3)), 0) # x^3
    assert np.isclose(newton(my_num, lambda x: power(x, 2, 1)), 1) # (x-1)^2
    assert np.isclose(newton(my_num, lambda x: power(x, 3, 1)), 1) # (x-1)^3
    assert np.isclose(newton(my_num, lambda x: power(x, 2, -1)), -1) # (x+1)^2
    assert np.isclose(newton(my_num, lambda x: power(x, 3, -1)), -1) # (x+1)^3

    assert np.isclose(newton(my_num, lambda x: power(x, 3, 0, 1)), -1) # x^3 + 1
    assert np.isclose(newton(my_num, lambda x: power(x, 2, 0, -1)), -1) # x^2 - 1, root=-1
    assert np.isclose(newton(my_num, lambda x: power(x, 3, 0, -1)), 1) # x^3 - 1

    my_num = ad.AutoDiff(0.5)
    assert np.isclose(newton(my_num, lambda x: power(x, 2, 0, -1)), 1) # x^2 - 1, root=1

    # Test guess is root
    my_num = ad.AutoDiff(0)
    assert np.isclose(newton(my_num, lambda x: power(x, 2)), 0)

    # Test zero derivative failure mode
    my_num = ad.AutoDiff(0)
    try:
        newton(my_num, lambda x: power(x, 2, 0, -1))
    except FloatingPointError:
        assert True

    # Test no root failure mode
    my_num = ad.AutoDiff(2)
    assert newton(my_num, lambda x: power(x, 2, 0, 1)) == "NO ROOT FOUND" # x^2 + 1


def test_polynomial():
    '''
        This function tests polynomials built from the power method
        The cases tested are:
            General polynomial with 3 roots
            polynomial with repeated root
            polynomial with imaginary roots
    '''

    def polynomial(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d


    # General polynomial
    func = lambda x: polynomial(x, 1, -2, -1, 2) # f(x) = (x+1)(x-1)(x-2)

    # Finding roots from the right
    my_num = ad.AutoDiff(-0.6)
    assert np.isclose(newton(my_num, func), -1)
    my_num = ad.AutoDiff(1.2)
    assert np.isclose(newton(my_num, func), 1)
    my_num = ad.AutoDiff(4)
    assert np.isclose(newton(my_num, func), 2)

    # Finding roots from the left
    my_num = ad.AutoDiff(-2)
    assert np.isclose(newton(my_num, func), -1)
    my_num = ad.AutoDiff(0.2)
    assert np.isclose(newton(my_num, func), 1)
    my_num = ad.AutoDiff(1.6)
    assert np.isclose(newton(my_num, func), 2)


    # Polynomial with repeated root
    func = lambda x: polynomial(x, 1, 0, -3, 2) # f(x) = (x-1)(x-1)(x+2)

    # Finding the roots from the right
    my_num = ad.AutoDiff(-1.1)
    assert np.isclose(newton(my_num, func), -2)
    my_num = ad.AutoDiff(2)
    assert np.isclose(newton(my_num, func), 1)

    # Finding the roots from the left
    my_num = ad.AutoDiff(-3)
    assert np.isclose(newton(my_num, func), -2)
    my_num = ad.AutoDiff(0)
    assert np.isclose(newton(my_num, func), 1)


    # Polynomial with imaginary roots
    func = lambda x: polynomial(x, 1, 0, 4, 0) # f(x) = x(x+2i)(x-2i)

    # Finding the root from the right
    my_num = ad.AutoDiff(1)
    assert np.isclose(newton(my_num, func), 0)

    # Finding the root from the left
    my_num = ad.AutoDiff(-1)
    assert np.isclose(newton(my_num, func), 0)


def test_trig():
    assert True


if __name__ == '__main__':
    #test_pow()
    test_polynomial()

