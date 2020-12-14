
import numpy as np
from time import time
from numba import jit

@njit
def oddevenSort(x):
    sorted = False
    while not sorted:
        sorted = True
        for i in range(0, len(x) - 1, 2):
            if x[i] > x[i + 1]:
                x[i], x[i + 1] = x[i + 1], x[i]
                sorted = False
        for i in range(1, len(x) - 1, 2):
            if x[i] > x[i + 1]:
                x[i], x[i + 1] = x[i + 1], x[i]
                sorted = False
    return x


times_quick = []    

for i in [7, 9, 10, 13, 16, 17, 19, 20, 21, 23, 24]:
#for i in [7]:
    array = np.load(f'./array_{i}.npy') 
    t0=time()
    oddevenSort(array.copy())
    t1 = time()
    times_quick.append(t1-t0)
    
    print("Time  ", t1-t0)
    
array_path = f'./times_quick.npy'
np.save(array_path, np.array(times_quick))
