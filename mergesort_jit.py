
import numpy as np
from time import perf_counter

@jit
def merge_sort(alist):
 
    if len(alist) > 1:
        mid = len(alist) // 2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        merge_sort(lefthalf)
        merge_sort(righthalf)

        i = 0
        j = 0 
        k = 0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] <= righthalf[j]:
                alist[k] = lefthalf[i]
                i = i+1
            else:
                alist[k]=righthalf[j]
                j = j+1
            k = k+1

        while i < len(lefthalf):
            alist[k] = lefthalf[i]
            i = i+1
            k = k+1

        while j < len(righthalf):
            alist[k] = righthalf[j]
            j = j+1
            k = k+1

    
    
times_merge_regular = []    
for i in [7, 9, 10, 13, 16, 17, 19, 20, 21]:
    array = np.load(f'Arrays/array_{i}.npy') 
    start = perf_counter()
    merge_sort(array)
    end = perf_counter()
    times_merge_regular.append(end-start)
    
array_path = 'times_merge_regular.npy'
np.save(array_path, np.array(times_merge_regular))
