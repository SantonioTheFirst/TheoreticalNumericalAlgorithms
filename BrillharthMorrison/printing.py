import numpy as np
# from numba import jit
def printing(a):
    print('\n{0}'.format(a))


def printIterationInfo(i : np.int, v : np.float, alpha : np.float, a : np.float, u : np.float) -> np.void: #####################
    print('\ni = {0}\n\nv_{0} = {1}, alpha_{0} = {2}, a_{0} = {3}, u_{0} = {4}'.format(i, v, alpha, a, u))

def printM(A : np.ndarray, b_2) -> np.void:
    for i in range(len(A)):
        print('{0} <-- {1}'.format(A[i], b_2[i]))
