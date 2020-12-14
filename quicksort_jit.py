import numpy as np
from time import perf_counter
from numba import jit

@jit
def partition(array, start, end):
    pivot = array[start]
    low = start + 1
    high = end

    while True:
        while low <= high and array[high] >= pivot:
            high = high - 1

        while low <= high and array[low] <= pivot:
            low = low + 1

        if low <= high:
            array[low], array[high] = array[high], array[low]
        else:
            break

    array[start], array[high] = array[high], array[start]
    return high

@jit
def quick_sort(array, start, end):
    if start >= end:
        return

    p = partition(array, start, end)
    quick_sort(array, start, p-1)
    quick_sort(array, p+1, end)

times_quick_jit = []
for i in [7, 9, 10, 13, 16, 17, 19, 20, 21, 23, 24]:
    array = np.load(f'Arrays/array_{i}.npy')
    start = perf_counter()
    quick_sort(array, 0, len(array)-1)
    end = perf_counter()
    times_quick_regular.append(end-start)

array_path = 'times_quick_jit.npy'
np.save(array_path, np.array(times_quick_jit))
