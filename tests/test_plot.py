import sys
sys.path.append(sys.path[0][:-5])

import os
import numpy as np
import autodiff.optimization as opt
from autodiff.autodiff import AutoDiff as ad
import autodiff.plot as plt


def test_plot():
    x = ad(1, [1., 0.])
    y = ad(1, [0., 1.])
    func = lambda x: x[0]**2 + x[1]**2
    
    output = opt.gradient_descent(func, [x, y], return_trace=True)
    plt.plot_2dtrajectory(func, output[3], output[1])
    
    assert 'trace.png' in os.listdir("./")
    os.remove("./trace.png")

if __name__ == '__main__':
    test_plot()
    
