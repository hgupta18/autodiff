import pytest
import func
import dual
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


if __name__ == '__main__':
    #test_dual_add()
    test_dual_mul()
