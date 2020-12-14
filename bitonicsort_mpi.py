
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def parallel_bitonic(array):

    start_timer = MPI.Wtime()
    phases = int(np.log2(size))
    array = np.sort(array, kind='mergesort')

    for i in range(phases):
        for j in range(i,-1,-1):
            if (rank >> (i + 1)) % 2 == 0 and  (rank >> j) % 2 == 0:
                array = CompareLow(j, array)
            elif (rank >> (i + 1)) % 2 != 0 and (rank >> j) % 2 != 0:
                array = CompareLow(j, array)
            else:
                array = CompareHigh(j, array)

    comm.Barrier()

    if rank == 0:
        for i in range(1, size):
            arr = comm.recv(source=i,tag=i)
            array= np.append(array, arr)
        end_timer = MPI.Wtime()
        return end_timer - start_timer
    else:
        comm.send(array,0,rank)

def CompareLow(j, array):

    comm.send(array, dest=rank ^ (1 << j), tag=0)
    receive = comm.recv(source=rank ^ (1 << j), tag=0)

    l = len(array)
    array = np.append(array,receive)
    array = np.sort(array, kind='mergesort')

    return array[:l]

def CompareHigh(j, array):

    receive = comm.recv(source=rank ^ (1 << j), tag=0)
    comm.send(array, dest=rank ^ (1 << j), tag=0)

    l = len(array)
    array = np.append(array,receive)
    array = np.sort(array, kind='mergesort')
    return array[l:]


times_bitonic = []    
for i in [7, 9, 10, 13, 16, 17, 19, 20, 21, 23, 24]:
    array = np.load(f'./array_{i}.npy') 
    if rank == 0:
        time = parallel_bitonic(array)
        times_bitonic.append(time)
    else:
        parallel_bitonic(array)
if rank == 0:
    array_path = f'./times_bitonic.npy'
    np.save(array_path, np.array(times_bitonic))
