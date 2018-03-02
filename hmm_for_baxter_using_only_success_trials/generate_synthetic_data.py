import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pylab as plt 
from scipy.spatial.distance import cdist
from numpy import array, zeros, argmin, inf, equal, ndim
import os
import ipdb 

#for the position synthetic traj generation

def fastdtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x)==1:
        x = x.reshape(-1,1)
    if ndim(y)==1:
        y = y.reshape(-1,1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:,1:] = cdist(x,y,dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    mindis = D1[-1, -1] / sum(D1.shape)
    return mindis

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

def _plot(traj, synthetic_data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(traj[:,1:].tolist(), color='gold', label='original data')
    for i in range(len(synthetic_data)):
        ax.plot(synthetic_data[i].tolist(), linestyle="dashed", color='gray', label='synthetic_traj')
    ax.legend()
    fig.show()

def run(df, csv_save_path):
    '''
    This function for generating num_data synthetic datas from the one-shot
    csv_path = '/media/vmrguser/DATA/Homlx/unsupervised_online_reaching_prediction/scripts/testing.csv'
    '''
    num_data = 5
    dis_thresthod = 2.0
    traj = df.values
    interested_data_fields = df.columns.values
    N,D = traj.shape
    A   = np.eye(N)
    x   = np.eye(N, k=-1)*-2.0
    A   = A + x
    x   = np.eye(N, k=-2) 
    A   = A + x
    _row = np.append([np.zeros(N-2)],[1, -2])
    A = np.vstack([A, _row])
    _row = np.append([np.zeros(N-1)],[1])
    A = np.vstack([A, _row])
    R_1 = np.linalg.inv(np.dot(A.T, A))
    y = np.amax(R_1, axis=0)
    y = np.matlib.repmat(y, N, 1)
    M = np.divide(R_1, y) * (1.0/N)

    traj_results = []
    synthetic_data = []
    for ind_D in range(num_data):
        theta = traj
        theta_k = np.random.multivariate_normal(np.zeros(N), R_1, D).T
        test_traj = theta + theta_k
        while fastdtw(test_traj, theta, dist=lambda x, y: np.linalg.norm(x - y, ord=1)) > dis_thresthod:
            print('.\n')
            theta_k = np.dot(M, theta_k)
            test_traj = theta + theta_k
        synthetic_traj = theta + theta_k
        df = pd.DataFrame(synthetic_traj.tolist(), columns=interested_data_fields)
        df.to_csv(os.path.join(csv_save_path, 'synthetic_' + str(ind_D) + '.csv'))
        synthetic_data.append(synthetic_traj)

    #plot 
#    _plot(traj, synthetic_data)
    
    
    
