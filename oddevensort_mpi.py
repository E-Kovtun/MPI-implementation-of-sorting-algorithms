import numpy as np
from mpi4py import MPI

t0 = MPI.Wtime()

comm = MPI.COMM_WORLD
rank = comm.Get_rank() 
size = comm.Get_size()  


def odd_even(data_local):
    
    #neighbours
    p1 = rank - 1 if rank % 2 == 0 else rank + 1
    odd_neigh = None if p1 < 0 or p1 >= size else p1
    
    p2 = rank + 1 if rank % 2 == 0 else rank - 1
    even_neigh = None if p2 < 0 or p2 >= size else p2
    
    neighbours_all = {0: even_neigh, 1: odd_neigh}
    
    for idx in range(size):
        neighbour_particular = neighbours_all[idx % 2]
        if neighbour_particular is None:
            continue
        data_local = iteration(data_local, neighbour_particular)
        
    return data_local


def iteration(data_local, neighbour_particular):
    
    neigh_data = np.empty(data_local.size, dtype=np.int)
    comm.Sendrecv(data_local, dest=neighbour_particular, recvbuf=neigh_data, source=neighbour_particular)
    
    data = np.sort(np.concatenate([data_local, neigh_data]))
    
    return data[:data_local.size] if rank < neighbour_particular else data[data_local.size:]


def sort(initial_array = [1]):
    
    if (initial_array.size % 2 != 0):
        raise ValueError("Should be even number of elements")
    if ((initial_array.size / size) % 1 != 0.0):
        raise ValueError("Number of elements in array should be divisible by #cores")
        
        
    data_local = np.array_split(initial_array, size)[rank]
    
    data_local = odd_even(data_local)
    
    final_sorted = None
    if rank == 0:
        final_sorted = np.empty([size, data_local.size], dtype=np.int)
    comm.Gather(data_local, final_sorted, root=0)
    
    
    if rank == 0:
        print(initial_array)
        print()
        print()
        
        print(final_sorted.flatten())
        t1 = MPI.Wtime()
        print()
        print()
        print()
        print("TIME ", t1-t0)
        print()
        print()
        print()
        
    #return (t1-t0), final_sorted.flatten()

    
    

# size should be even and divisible on size    (comm.Get_size())
#! size should be>1, otherwise array will not be sorted, use regular version :-(
N = 2**8
q = np.random.randint(low=0,high=N,size=N)
#q =np.array([10,2,30,1,0,120,-1,0])
sort(q)

#array = np.load('Arrays/array_21.npy') 

 
#sort(array)
