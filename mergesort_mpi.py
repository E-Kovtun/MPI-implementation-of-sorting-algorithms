
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
status = MPI.Status()

def parallel_merge(array):
    
    start = MPI.Wtime()   # counting the time
    N = len(array)
    
    unsorted_array = np.zeros(N, dtype="int")   # create null array for unsorted initial array
    local_array = np.zeros(int(N / size), dtype="int")   # create null array for dividing by process
    
    # create 2 arrays for merging sub-arrays
    tmp_array = np.zeros(int(N / size), dtype="int") 
    remain_array = np.zeros(2 * int(N / size), dtype="int")

    if rank == 0:
        local_array = array 

    comm.Scatter(unsorted_array, local_array, root = 0)

    local_array = np.sort(local_array, kind='mergesort') # sorting arrays in each process 

    step = size / 2
    while (step >= 1):
        if (rank >= step and rank < step * 2):
            comm.Send(local_array, rank - step, tag = 0)    # on the fisrt half of processes
        elif (rank < step):
            tmp_array = np.zeros(local_array.size, dtype="int")
            remain_array = np.zeros(2 * local_array.size, dtype="int")
            comm.Recv(tmp_array, rank + step, tag = 0)    # on the second half of processes
            i = 0    # local_array counter
            j = 0    # tmp_array counter
            
            for k in range (0, 2 * local_array.size):    # merging sub-arrays
                if (i >= local_array.size): 
                    remain_array[k] = tmp_array[j]
                    j += 1
                elif (j >= local_array.size):
                    remain_array[k] = local_array[i]
                    i += 1
                elif (local_array[i] > tmp_array[j]):
                    remain_array[k] = tmp_array[j]
                    j += 1
                else:
                    remain_array[k] = local_array[i]
                    i += 1        

            local_array = remain_array
        step = step / 2
    end = MPI.Wtime()
    timing = end - start
    
    sorted_array = local_array   
    if (rank  == 0):
        return timing #, sorted_array

    
times_merge = []    
for i in [7, 9, 10, 13, 16, 17, 19, 20, 21, 23, 24]:
    array = np.load(f'Arrays/array_{i}.npy') 
    if rank == 0:
        time = parallel_merge(array)
        times_merge.append(time)
    else:
        parallel_merge(array)
if rank == 0:
    array_path = 'times_merge.npy'
    np.save(array_path, np.array(times_merge))
    
