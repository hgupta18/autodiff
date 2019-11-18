import sys
sys.path.append(sys.path[0][:-5])

import pytest
from autodiff.autodiff import AutoDiff

def test_add_objects():
    error_message = "Problem with test_add_objects"
    AD1 = AutoDiff(1,2)
    AD2 = AutoDiff(3,4)
    AD3 = AD1 + AD2
    assert AD3.val == 4, error_message
    assert AD3.der == 6, error_message

def test_add_constant():
    error_message = "Problem with test_add_constant"
    AD1 = AutoDiff(1,2)
    AD2 = AD1 + 4
    assert AD2.val == 5, error_message
    assert AD2.der == 2, error_message

# def test_add_string():

def test_subtract_objects():
    error_message = "Problem with test_subtract_objects"
    AD1 = AutoDiff(5,10)
    AD2 = AutoDiff(3,4)
    AD3 = AD1 - AD2
    assert AD3.val == 2, error_message
    assert AD3.der == 6, error_message
    
    AD4 = AD2 - AD1
    assert AD4.val == -2, error_message
    assert AD4.der == -6, error_message


def test_subtract_constant():
    error_message = "Problem with test_subtract_constant"
    AD1 = AutoDiff(5,10)
    AD2 = AD1 - 3
    assert AD2.val == 2, error_message
    assert AD2.der == 10, error_message

def test_multiply_objects():
    error_message = "Problem with test_multiply_objects"
    AD1 = AutoDiff(5,15)
    AD2 = AutoDiff(2,3)
    AD3 = AD1*AD2
    assert AD3.val == 10, error_message
    assert AD3.der == 45, error_message

def test_multiple_constant():
    error_message = "Problem with test_multiply_objects"
    AD1 = AutoDiff(5,15)
    AD2 = AD1*2
    assert AD2.val == 10, error_message
    assert AD2.der == 30, error_message

# def test_divide_objects():
#     AD1 = AutoDiff(15,45)
#     AD2 = AutoDiff(30,90)
#     AD3 = AD1/AD2
    # assert AD3.val == 0.5
    # assert AD3.der == 
def test_power():
    AD1 = AutoDiff(2,3)
    AD2 = AD1**5
    assert AD2.val == 32
    assert AD2.der == 240

def test_negation():
    AD1 = AutoDiff(2,3)
    AD2 = -AD1
    assert AD2.val == -2
    assert AD2.der == -3

# def test_sin():
    # AD.
