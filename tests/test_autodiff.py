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

def test_radd():
    AD1 = AutoDiff(1,2)
    AD2 = 4 + AD1
    assert AD2.val == 5
    assert AD2.der == 2

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

def test_rsub():
    AD1 = AutoDiff(5,10)
    AD2 = 8 - AD1
    assert AD2.val == 3
    assert AD2.der == -10

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

def test_rmul():
    AD1 = AutoDiff(5,15)
    AD2 = 2*AD1
    assert AD2.val == 10
    assert AD2.der == 30

def test_divide_objects():
    AD1 = AutoDiff(15,45)
    AD2 = AutoDiff(30,90)
    AD3 = AD1/AD2
    assert AD3.val == 0.5
    assert AD3.der == (45*30-15*90)/(30**2)
    AD4 = AD2.__rtruediv__(AD1)
    assert AD4.val == 0.5
    assert AD4.der == (45*30-15*90)/(30**2) 

def test_divide_constant():
    AD1 = AutoDiff(15,45)
    AD3 = AD1/3
    assert AD3.val == 5
    assert AD3.der == (1/3)*(45)

def test_rtruediv():
    AD1 = AutoDiff(3,5)
    AD3 = 15/AD1
    assert AD3.val == 5
    assert AD3.der == (-15*5)/(3**2)  

def test_power_objects():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff(3,4)
    AD3 = AD1**AD2
    assert AD3.val == 8
    assert AD3.der == 3*(2**(3-1))*3

def test_power_constant():
    AD1 = AutoDiff(2,3)
    AD2 = AD1**5
    assert AD2.val == 32
    assert AD2.der == 240

def test_rpow_objects():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff(3,4)
    AD3 = AD1.__rpow__(AD2)
    assert AD3.val == 3**2
    assert AD3.der == 2 * (3**(2-1)) * 4

def test_rpow_constant():
    AD1 = AutoDiff(2,3)
    AD2 = 5**AD1
    assert AD2.val == 5**2
    assert AD2.der == 3 * (5**2) * np.log(5)


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

def test_arcsine():
    AD1 = AutoDiff(0.5,3)
    AD2 = AutoDiff.arcsine(AD1)
    assert AD2.val == np.arcsin(0.5)
    assert AD2.der == 3/((1 - 0.5**2)**(1/2))

def test_arccosine():
    AD1 = AutoDiff(0.5,3)
    AD2 = AutoDiff.arccosine(AD1)
    assert AD2.val == np.arccos(0.5)
    assert AD2.der == -3/((1 - 0.5**2)**(1/2))

def test_arctangent():
    AD1 = AutoDiff(0.5,3)
    AD2 = AutoDiff.arctangent(AD1)
    assert AD2.val == np.arctan(0.5)
    assert AD2.der == 3/(0.5**2 + 1)

def test_sinh():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff.sinh(AD1)
    assert AD2.val == np.sinh(2)
    assert AD2.der == 3*np.cosh(2)

def test_cosh():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff.cosh(AD1)
    assert AD2.val == np.cosh(2)
    assert AD2.der == 3*np.sinh(2)

def test_tanh():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff.tanh(AD1)
    assert AD2.val == np.tanh(2)
    assert AD2.der == 3*1/(np.cosh(2)**2)

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

def test_expm():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff.expm(AD1, 10)
    assert AD2.val == 10 ** 2
    assert AD2.der == 3 * (10 ** 2) * np.log(10)

def test_logistic():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff.logistic(AD1)
    assert AD2.val == 1 / (1 + np.exp(-2))
    assert AD2.der == 3*np.exp(-2)/(1+np.exp(-2))**2

def test_sqrt():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff.sqrt(AD1)
    assert AD2.val == 2 ** (1/2)
    assert AD2.der == 3*(1/2)*(2**(-1/2))

def test_str():
    AD1 = AutoDiff(2,3)
    assert AutoDiff.__str__(AD1) == "AutoDiff(2,3)"

def test_repr():
    AD1 = AutoDiff(2,3)
    assert AutoDiff.__repr__(AD1) == "AutoDiff(2,3)"

def test_eq():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff(3,3)
    AD3 = AutoDiff(2,-4)
    assert AD1.__eq__(AD2) == False
    assert AD1.__eq__(AD3) == True
    assert AD1.__eq__(2) == True
    assert AD1.__eq__(3) == False

def test_ne():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff(3,3)
    AD3 = AutoDiff(2,-4)
    assert AD1.__ne__(AD2) == True
    assert AD1.__ne__(AD3) == False
    assert AD1.__ne__(2) == False
    assert AD1.__ne__(3) == True

def test_gt():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff(3,3)
    assert AD1.__gt__(2) == False
    assert AD1.__gt__(0) == True
    assert AD1.__gt__(AD2) == False
    assert AD2.__gt__(AD1) == True

def test_ge():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff(3,3)
    assert AD1.__ge__(2) == True
    assert AD1.__ge__(4) == False
    assert AD1.__ge__(AD2) == False
    assert AD2.__ge__(AD1) == True

def test_lt():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff(3,3)
    assert AD1.__lt__(2) == False
    assert AD1.__lt__(4) == True
    assert AD1.__lt__(AD2) == True
    assert AD2.__lt__(AD1) == False

def test_le():
    AD1 = AutoDiff(2,3)
    AD2 = AutoDiff(3,3)
    assert AD1.__le__(2) == True
    assert AD1.__le__(4) == True
    assert AD1.__le__(AD2) == True
    assert AD2.__le__(AD1) == False