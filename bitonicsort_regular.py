import numpy as np
from time import perf_counter


def bitonic_sort(array):
    bitonic_split(array, 0, len(array), True)

def bitonic_split(array, start, end, bit):
    if end > 1:
        k = end // 2
        bitonic_split(array, start, k, not bit)
        bitonic_split(array, start + k, end-k, bit)
        bitonic_merge(array, start, end, bit)

def compare_exchange(array, i, j, bit):
    if (bit and array[i] > array[j]) or (not bit and array[i] <= array[j]):
        array[i], array[j] = array[j], array[i]

def bitonic_merge(array, start, end, bit):
    if end > 1:
        k = number_of_steps(end)
        for i in range(start, start + end-k):
            compare_exchange(array, i, i + k, bit)
        bitonic_merge(array, start, k, bit)
        bitonic_merge(array, start + k, end-k, bit)

def number_of_steps(n):
    k = 1
    while k>0 and k<n:
        k = k<<1
    return k>>1
    
times_bitonic_regular = []    
for i in [7, 9, 10, 13, 16, 17, 19, 20, 21, 23, 24]:
    array = np.random.random(4)
    start = perf_counter()
    bitonic_sort(array)
    end = perf_counter()
    times_bitonic_regular.append(end-start)
    
array_path = 'times_bitonic_regular.npy'
np.save(array_path, np.array(times_bitonic_regular))