import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors


def plot_MT(acc_cost_matrix, idx=[], values=False):
    if not values:
        fig2, ax2 = plt.subplots(figsize=(18, 16))
        ax2.set_title("Cost Matrix")
        color_map = cm.get_cmap('binary')
        ax2.matshow(acc_cost_matrix.T, cmap=matplotlib.colors.ListedColormap(color_map(np.linspace(0,0.01))))
        ax2.set_ylim(ax2.get_ylim()[::-1])
        ax2.xaxis.tick_bottom()
        for i in range(acc_cost_matrix.shape[0]):
            for j in range(acc_cost_matrix.shape[1]):
                c = acc_cost_matrix[i, j]
                if c not in idx:
                    ax2.text(i, j, "{:.4f}".format(c), color='g', va='center', ha='center')
                else:
                    ax2.text(i, j, "{:.4f}".format(c), color='r', va='center', ha='center')
    else:
        cm2 = np.ones_like(acc_cost_matrix)
        for i in range(cm2.shape[0]):
            for j in range(cm2.shape[1]):
                if acc_cost_matrix[i,j] in idx:
                    cm2[i,j] = -1
        fig2, ax2 = plt.subplots(figsize=(18, 16))
        ax2.set_title("Cost Matrix")
        color_map = cm.get_cmap('Pastel1')
        ax2.matshow(cm2.T, cmap=matplotlib.colors.ListedColormap(color_map(np.linspace(0.1,0.3))))
        ax2.set_ylim(ax2.get_ylim()[::-1])
        ax2.xaxis.tick_bottom()


def forbidden_list_gen(xsi, mat):    
    m, n = mat.shape
    forbidden_list = []
    seed = xsi
    for col in range(m):
        start = seed + col*(m+1)
    #    print(start, end=" ")
        end = (col+1)*m
    #    print(end)
        forbidden_list.extend(np.arange(start, end+1, 1))
    seed2 = (m**2)-seed+1
    for col in range(m, 0, -1):
        start = seed2 - (m-col)*(m+1)
    #    print(start, end=" ")
        end = m**2 - (m-col+1)*(m)+1
    #    print(end)
        forbidden_list.extend(np.arange(start, end-1, -1))
    return forbidden_list

def dtw_random_path(xsi, mat, prob=[0.33, 0.33, 0.34]):
    """
    Parameters
    ----------
    xsi : Range
    mat : Cost matrix indices
    prob : [right_prob, left_prb, diag_prob] The default is [0.33, 0.33, 0.34].
    """
    m, n = mat.shape
    i = 0
    j = 0
    path = [mat[i, j]]
    forbidden_list = forbidden_list_gen(xsi, mat)
    while i!=m-1 and j!=n-1:
        step = np.random.choice([1,2,3], p=prob)
        if step==1 and i+1 < m: #move right
            if mat[i+1, j] in forbidden_list:
                continue
            i += 1
        if step==2 and j+1 < m: #move up
            if mat[i, j+1] in forbidden_list:
                continue
            j += 1
        if step==3 and i+1 < m and j+1<m: #move diag
            i += 1
            j += 1
        path.append(mat[i, j])
    if i==m-1:
        while j!=n-1:
            j+=1
            path.append(mat[i, j])
    if j==n-1:
        while i!=m-1:
            i+=1
            path.append(mat[i, j])
    return path

def path_conversion(p, mat):
    dict_el = {}
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            dict_el[mat[i,j]]=(i,j)
    dtw_path0 = []
    dtw_path1 = []
    for el in p:
        idx = dict_el[el]
        dtw_path0.append(idx[0])
        dtw_path1.append(idx[1])
    return (np.array(dtw_path0),np.array(dtw_path1))

def gen_all_paths(m, n, start=None, M=None):
    """
    Generator that produces all possible paths between (1, 1) and (m, n), using
    only north, northeast, or east steps. Each path is represented as a (m, n)
    numpy array with ones indicating the path.

    Parameters
    ----------
    m, n : int, int
        Northeast corner coordinates.
    """
    if start is None:
        start = [0, 0]
        M = np.zeros((m, n))

    i, j = start
    M[i, j] = 1
    ret = []

    if i == m-1 and j == n-1:
        yield M
    else:
        if i < m - 1:
            # Can use yield_from starting from Python 3.3.
            for mat in gen_all_paths(m, n, (i+1, j), M.copy()):
                yield mat
        if i < m-1 and j < n-1:
            for mat in gen_all_paths(m, n, (i+1, j+1), M.copy()):
                yield mat
        if j < n-1:
            for mat in gen_all_paths(m, n, (i, j+1), M.copy()):
                yield mat