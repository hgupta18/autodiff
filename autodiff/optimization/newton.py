import sys
sys.path.append(sys.path[0][:-21])

import numpy as np
from autodiff.new_autodiff import AutoDiff as ad


def newton(f, x):
    pass


if __name__ == '__main__':
    xs = [1]
    ys = [1]
    x = ad(1, [1.,0.,0.])
    y = ad(1, [0.,1.,0.])
    z = ad(1, [0.,0.,1.])

    fn = lambda x: (x[0].cos()+1)**2 + x[1]**2 + (x[2]-2)**2
    print(newton(f, [x,y,z]))
