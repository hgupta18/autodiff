import sys
import numpy as np
sys.path.append(sys.path[0][:-5])

import pytest
from autodiff.autodiff import AutoDiff

def test_add_objects():
    AD1 = AutoDiff(1,2)
    AD2 = AutoDiff(3,4)
    AD3 = AD1 + AD2
    assert AD3.val == 4
    assert AD3.der == 6

def test_add_constant():
    AD1 = AutoDiff(1,2)
    AD2 = AD1 + 4
    assert AD2.val == 5
    assert AD2.der == 2

# def test_add_string():

def test_subtract_objects():
    AD1 = AutoDiff(5,10)
    AD2 = AutoDiff(3,4)
    AD3 = AD1 - AD2
    assert AD3.val == 2
    assert AD3.der == 6
    
    AD4 = AD2 - AD1
    assert AD4.val == -2
    assert AD4.der == -6


def test_subtract_constant():
    AD1 = AutoDiff(5,10)
    AD2 = AD1 - 3
    assert AD2.val == 2
    assert AD2.der == 10

def test_multiply_objects():
    AD1 = AutoDiff(5,15)
    AD2 = AutoDiff(2,3)
    AD3 = AD1*AD2
    assert AD3.val == 10
    assert AD3.der == 45

def test_multiple_constant():
    AD1 = AutoDiff(5,15)
    AD2 = AD1*2
    assert AD2.val == 10
    assert AD2.der == 30

def test_divide_objects():
    AD1 = AutoDiff(15,45)
    AD2 = AutoDiff(30,90)
    AD3 = AD1/AD2
    assert AD3.val == 0.5
    assert AD3.der == (45*30-15*90)/(30**2)

def test_divide_constant():
    AD1 = AutoDiff(15,45)
    AD3 = AD1/3
    assert AD3.val == 5
    assert AD3.der == (1/3)*(45)

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

def test_sin():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff.sin(AD1)
    assert AD2.val == np.sin(2)
    assert AD2.der == 3*np.cos(2)

def test_cos():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff.cos(AD1)
    assert AD2.val == np.cos(2)
    assert AD2.der == -3*np.sin(2)

def test_tan():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff.tan(AD1)
    assert AD2.val == np.tan(2)
    assert AD2.der == 3/(np.cos(2)**2)

def test_ln():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff.ln(AD1)
    assert AD2.val == np.log(2)
    assert AD2.der == (1/2)*3

def test_log():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff.log(AD1,5)
    assert AD2.val == np.log(2)/np.log(5)
    assert AD2.der == (1/(2*np.log(5)))*3

def test_exp():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff.exp(AD1)
    assert AD2.val == np.exp(2)
    assert AD2.der == 3*np.exp(2)

def test_sqrt():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff.sqrt(AD1)
    assert AD2.val == 2 ** (1/2)
    assert AD2.der == 3*(1/2)*(2**(-1/2))