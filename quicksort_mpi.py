import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
status = MPI.Status()

def parallel_quick(array):
    start = MPI.Wtime()
    
    N = len(array)
    HAS = 1
    HASNOT = 0
    local_array = None
    local_tmp = None
    local_tmp_size = np.zeros(1,dtype="int")
    if rank == 0:
        local_array = array

    distance = size / 2
    while (distance >= 1):
        if (rank % distance == 0 and (rank / distance) % 2 == 0):
            if (local_array is not None):
                if local_array.size == 1 or np.unique(local_array).size == 1:
                    comm.Send(local_array[0], dest = rank + distance, tag = HASNOT)
                else:
                    local_tmp = local_array[local_array > np.median(local_array)]
                    comm.Send(np.full(shape = 1, fill_value = local_tmp.size, dtype="int"), dest = rank + distance, tag = HAS)
                    comm.Send(local_tmp, dest = rank + distance, tag = HAS)
                    local_array = local_array[local_array <= np.median(local_array)]
            else:
                comm.Send(np.zeros(1,dtype="int"), rank + distance, tag = HASNOT)
        elif (rank % distance == 0 and (rank / distance) % 2 == 1):
            comm.Recv(local_tmp_size, source = rank - distance, tag = MPI.ANY_TAG, status = status)
            if status.Get_tag() == HASNOT:
                continue
            else:
                local_array = np.zeros(local_tmp_size[0], dtype="int")
                comm.Recv(local_array, source = rank - distance, tag = MPI.ANY_TAG, status = status)
        distance /= 2

    local_array = np.sort(local_array, kind='quicksort')
    if rank != 0:
        comm.send(len(local_array), dest=0, tag=10)
        comm.Send(local_array, dest=0, tag=20)
    
    if rank == 0:
        final_array = local_array
        for r in range(1, size):
            recv_size = comm.recv(source=r, tag=10)
            recv_array = np.empty(recv_size, dtype='int')
            comm.Recv(recv_array, source=r, tag=20)
            final_array = np.hstack((final_array, recv_array))
            
        end = MPI.Wtime()
        time = end-start
        return time

times_quick = []    
for i in [7, 9, 10, 13, 16, 17, 19, 20, 21, 23, 24]:
    array = np.load(f'Arrays/array_{i}.npy') 
    if rank == 0:
        time = parallel_quick(array)
        times_quick.append(time)
    else:
        parallel_quick(array)
if rank == 0:
    array_path = 'times_quick.npy'
    np.save(array_path, np.array(times_quick))
    
