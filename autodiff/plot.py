import sys
import numpy as np
#from autodiff import AutoDiff as ad 
from autodiff.autodiff import AutoDiff as ad
from matplotlib import pyplot as plt

#import optimization as opt
#import root_finding as rf

def _ad_to_scalar(points):
    '''
        Private function to convert list of AutoDiff objects to floats
    '''
    scalar_points = []
    for p in points:
        scalar_points.append([p[0].val, p[1].val])
    return np.array(scalar_points)


def _func_grid_points(func, X, Y):
    func_pos = np.zeros(X.shape)
    root = isinstance(func([X[0][0], Y[0][0]]), list)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if(root):
                func_pos[i][j] = func([X[i][j], Y[i][j]])[0]
            else:
                func_pos[i][j] = func([X[i][j], Y[i][j]])

    return func_pos
    

def plot_2dtrajectory(func, points, converged=False, spacing=0.05, levels=20):
    '''
        INPUTS:
            func: callable
                  function being minimized/root-found

            points: ndarray of AutoDiff objects

            converged: boolean, optional (default = False)
                       If True, plots the last point as green star to show convergence

            levels: int, optional (default = 20)
                    Number of levels to use for contour plot

        OUTPUTS:
            figure: shows plot and saves it to current directory

        NOTES:
            pass
    '''
    # Expands plot borders around min and max elements in trace
    x_min = min(points[:,0])
    x_max = max(points[:,0])
    x_dist = x_max - x_min
    x_min -= x_dist/10
    x_max += x_dist/10

    y_min = min(points[:,1])
    y_max = max(points[:,1])
    y_dist = y_max - y_min
    y_min -= y_dist/10
    y_max += y_dist/10

    # Created grid used for contour plots
    X, Y = np.mgrid[x_min.val: x_max.val :spacing, y_min.val :y_max.val :spacing]
    pos = np.dstack((X, Y))

    # Converts output to points easier to plot
    scalar_points = _ad_to_scalar(points)
    func_grid_points = _func_grid_points(func, X, Y)

    fig, ax = plt.subplots()
    ax.contourf(X, Y, func_grid_points, cmap='Reds', levels=levels)
    ax.scatter(scalar_points[:,0], scalar_points[:,1])
    if(converged):
        ax.scatter(scalar_points[:,0][-1], scalar_points[:,1][-1], marker='*', color='k',
                   s=70, label="Converged Point")
    plt.savefig("./trace.png")
    plt.close(fig=fig)


if __name__ == '__main__':
    x = ad(-1, [1., 0.])
    y = ad(1, [0., 1.])

    fn = lambda x: 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    #fn = lambda x: (x[0]-1.4)**2 + x[1]**2
    #fn = lambda x: [x[0]**3 + x[1]**3]

    output = opt.conjugate_gradient(fn, [x,y], return_trace=True, step_size=0.00004,
                                    max_iter=500)
  
    #print(rf.newton(fn, [x, y], return_trace=True))
    #output = rf.newton(fn, [x, y], return_trace=True)
    print(output[-1])
    print(output[-1].shape)
    plot_2dtrajectory(fn, output[-1], output[1], spacing=0.001, levels=100)

