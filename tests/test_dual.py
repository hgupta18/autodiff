import sys
sys.path.append('../')

import pytest
for p in sys.path:
    print(p)
#import makeADifference
import makeADifference.dual as dual
import numpy as np

def test_dual_add():
    my_num = dual.Dual(1,1)
    assert my_num+1 == dual.Dual(2, 1)
    assert 1+my_num == dual.Dual(2, 1)
    assert my_num + dual.Dual(2,2) == dual.Dual(3,3)

def test_dual_mul():
    my_num = dual.Dual(2, 2)
    assert my_num*2 == dual.Dual(4,4)
    assert 2*my_num == dual.Dual(4,4)
    assert my_num*dual.Dual(2,3) == dual.Dual(4,10)

def test_repr():
    my_num = dual.Dual(2, 3)
    assert repr(my_num) == "REAL: {}\nDUAL: {}".format(my_num.real, my_num.dual)

def test_str():
    my_num = dual.Dual(2, 3)
    assert str(my_num) == "{} + e{}".format(my_num.real, my_num.dual)

