//Advanced odd-even sort implementation
// The only restriction is that each MPI process can only send messages to its neighbor processes. The number of elements sent in each message can also be arbitrary.
// For instance, MPI process with rank 6 can only send messages to process with rank 5 and process with rank 7.
// You are free to use any sorting algorithm within an MPI process.
// Advanced version should achieve better performance than basic version.
//
//First each process do quick sort and then two process combine their data and do merge sort
//check if it is sorted by 
//1. process has a swap when it is doing quick sort 
//2. when doing merge sort, there is a order that not follows all the data in one process are smaller than all data in the other process  
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ROOT 0

void swap(int* a, int* b){
	int temp;
	temp = *a;
	*a = *b;
	*b = temp;
	return;
}

//quicksort
void quicksort(int *data, int left, int right, int *sorted){
	int pivot, i, j;
	
	if(left >= right){
		return;
	}
	
	pivot = data[left];
	i = left + 1;
	j = right;
	
	while(1){
		while(i <= right){
			if(data[i] > pivot){
				break;
			}
			i++;
		}
		
		while(j > left){
			if(data[j] < pivot){
				break;
			}
			j--;
		}
		
		if(i > j){
			break;
		}
		
		swap(&data[i], &data[j]);
		if(i != j){
			*sorted = 0;
		}
	}
	
	swap(&data[left], &data[j]);
	if(left != j){
		*sorted = 0;
	}
	
	quicksort(data, left, j-1, sorted);
	quicksort(data, j+1, right, sorted);
}

//merge two sorted array and give them new sort
void merge(int *data, int *left, int sizeL, int *right, int sizeR, int* sorted){
	int i, j, k;
	i = j = k = 0;
	
	//combine and sort
	while(i < sizeL && j < sizeR){
		if(left[i] < right[j]){
			data[k] = left[i];
			i++;
		}
		else{
			data[k] = right[j];
			j++;
			*sorted = 0;
		}
		k++;	
	}
	
	while(i < sizeL){
		data[k] = left[i];
		i++;
		k++;
		*sorted = 0;
	}
	
	while(j < sizeR){
		data[k] = right[j];
		j++;
		k++;
	}
		
	//assign new sort
	for(i = 0; i < sizeL; i++){
		left[i] = data[i];
	}
	for(j = 0; j < sizeR; j++){
		right[j] = data[i+j];
	}
	return;
}

int main (int argc, char *argv[]) {
        //declare variables  chunk2 and mysize2 for another process
		int rank, size, rc, i, start, end, *chunk, mysize, totalSize, k, ierr, *data, *chunk2, mysize2;
		double TComm = 0, TIO = 0, Ttemp, TStart;
		MPI_Status status;

		//initialize MPI
        rc = MPI_Init(&argc, &argv);
		TStart = MPI_Wtime();
		//check if initialize success
		if(rc != MPI_SUCCESS){
			printf("Error starting MPI program. Terminating.\n");
			MPI_Abort(MPI_COMM_WORLD, rc);
		}
		
		//get size = total processes # & rank = each rank #
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		//wrong arguments #
        if (argc < 3) {
                if (rank == ROOT) {
                        fprintf(stderr, "Insufficient args\n");
                        fprintf(stderr, "Usage: %s N input_file", argv[0]);
                }
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Finalize();
                return 0;
        }

		//N # of input inName* inputfile outName outputfile
        const int N = atoi(argv[1]);
        const char *inName = argv[2];
		const char *outName = argv[3];
		data = malloc((2 * N / size + 1) * sizeof(MPI_INT));
	
        // Part 1: Read file
        /* Note: You should deal with cases where (N < size) in Homework 1 */
		MPI_File in, out;
		Ttemp = MPI_Wtime();
		ierr = MPI_File_open(MPI_COMM_WORLD, inName, MPI_MODE_RDONLY, MPI_INFO_NULL, &in);//printf("58 OK\n");
		if (ierr) {
			if (rank == 0) fprintf(stderr, "Couldn't open file %s\n", inName);
			MPI_Finalize();
			exit(2);
		}
		ierr = MPI_File_open(MPI_COMM_WORLD, outName, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &out);//printf("64 OK\n");
		if (ierr) {
			if (rank == 0) fprintf(stderr, "Couldn't open file %s\n", outName);
			MPI_Finalize();
			exit(2);
		}
		TIO += MPI_Wtime() - Ttemp;
		totalSize = size;
		//N > =size
		if(N >= size){
			//initialize
			start = N / size *rank;
			if(rank != size-1){
				end = N / size * (rank + 1) - 1;
			}
			else{
				end = N - 1;
			}
			mysize = end - start + 1;
			chunk = malloc((end - start + 1 + 1) * sizeof(MPI_INT));
			chunk2 = malloc((N / size * 2) * sizeof(MPI_INT));
			Ttemp = MPI_Wtime();
            MPI_File_read_at_all(in, (MPI_Offset)(start*sizeof(int)), chunk, mysize, MPI_INT, MPI_STATUS_IGNORE);
			TIO += MPI_Wtime() - Ttemp;
		}
		//N < size
        else if(N < size) {
                start = rank;
				if(rank < N){
					mysize = 1;
				}
				else{
					mysize = 0;
				} 
				chunk = malloc(2* sizeof(int));
				chunk2 = malloc(2 * sizeof(int));
				end = rank;
				Ttemp = MPI_Wtime();
				MPI_File_read_at_all(in, (MPI_Offset)(start*sizeof(int)), chunk, mysize, MPI_INT, MPI_STATUS_IGNORE);
				TIO += MPI_Wtime() - Ttemp;
				size = N;
        }
		Ttemp = MPI_Wtime();
		MPI_File_close(&in);
		TIO += MPI_Wtime() - Ttemp;

		//Part 2: odd-even sort
		int sorted = 0;
		if(rank < size){
			while(!sorted){
				sorted = 1;
				quicksort(chunk, 0, mysize - 1, &sorted);
				//odd sort
				i = 0;
				if(rank % 2 == 1){
					Ttemp = MPI_Wtime();
					MPI_Send(&mysize, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
					MPI_Send(chunk, mysize, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
					MPI_Recv(chunk, mysize, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
					TComm += MPI_Wtime() - Ttemp;
					
				}
				if(rank != size-1 && rank % 2 == 0){
					Ttemp = MPI_Wtime();
					MPI_Recv(&mysize2, 1, MPI_INT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
					MPI_Recv(chunk2, mysize2, MPI_INT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
					TComm += MPI_Wtime() - Ttemp;
					merge(data, chunk, mysize, chunk2, mysize2, &sorted);
					Ttemp = MPI_Wtime();
					MPI_Send(chunk2, mysize2, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
					TComm += MPI_Wtime() - Ttemp;
				}

				//even sort
				i = 0;
				if(rank % 2 == 0){
					Ttemp = MPI_Wtime();
					MPI_Send(&mysize, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
					MPI_Send(chunk, mysize, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
					MPI_Recv(chunk, mysize, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
					TComm += MPI_Wtime() - Ttemp;
					
				}
				if(rank != size-1 && rank % 2 == 1){
					Ttemp = MPI_Wtime();
					MPI_Recv(&mysize2, 1, MPI_INT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
					MPI_Recv(chunk2, mysize2, MPI_INT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
					TComm += MPI_Wtime() - Ttemp;
					merge(data, chunk, mysize, chunk2, mysize2, &sorted);
					Ttemp = MPI_Wtime();
					MPI_Send(chunk2, mysize2, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
					TComm += MPI_Wtime() - Ttemp;
				}

				//sorted
				if(size >= 2){
					if(rank == 0){
						Ttemp = MPI_Wtime();
						MPI_Send(&sorted, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
						MPI_Recv(&k, 1, MPI_INT, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
						TComm += MPI_Wtime() - Ttemp;
						if(k == 0) sorted = k;
					}
					else if(rank == size - 1){
						Ttemp = MPI_Wtime();
						MPI_Recv(&k, 1, MPI_INT, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
						TComm += MPI_Wtime() - Ttemp;
						if(k == 0) sorted = k;
						Ttemp = MPI_Wtime();
						MPI_Send(&sorted, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
						TComm += MPI_Wtime() - Ttemp;
					}
					else{
						Ttemp = MPI_Wtime();
						MPI_Recv(&k, 1, MPI_INT, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
						TComm += MPI_Wtime() - Ttemp;
						if(k == 0) sorted = k;
						Ttemp = MPI_Wtime();
						MPI_Send(&sorted, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
						MPI_Recv(&k, 1, MPI_INT, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
						TComm += MPI_Wtime() - Ttemp;
						if(k == 0) sorted = k;
						Ttemp = MPI_Wtime();
						MPI_Send(&sorted, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
						TComm += MPI_Wtime() - Ttemp;
					}
				}
			}
		}
		// write file
		Ttemp = MPI_Wtime();
		MPI_File_write_at(out, (MPI_Offset)(start*sizeof(int)), chunk, mysize, MPI_INT, MPI_STATUS_IGNORE);
		TIO += MPI_Wtime() - Ttemp;
		free(chunk);
		if(totalSize != size){
			if(rank == size - 1){
				Ttemp = MPI_Wtime();
				MPI_Send(&sorted, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
				TComm += MPI_Wtime() - Ttemp;
			}
			else if(rank >= size && rank != totalSize-1){
				Ttemp = MPI_Wtime();
				 MPI_Recv(&sorted, 1, MPI_INT, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				 MPI_Send(&sorted, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
				 TComm += MPI_Wtime() - Ttemp;
			}
			else if(rank == totalSize-1){
				Ttemp = MPI_Wtime();
				 MPI_Recv(&sorted, 1, MPI_INT, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				 TComm += MPI_Wtime() - Ttemp;
			}
		}
		Ttemp = MPI_Wtime();
		MPI_File_close(&out);
		TIO += MPI_Wtime() - Ttemp;
		if(rank == 0){
			printf("total time:%lf\ncomputing time:%lf\ncommunication time:%lf\nIO time:%lf\n", MPI_Wtime() - TStart, MPI_Wtime()-TStart-TComm-TIO, TComm, TIO);
		}
		if(sorted){
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Finalize();
		}
		else{
			printf("rank: %d   sorted:  ", rank, sorted);
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Finalize();
			return 2;
		}
        return 0;
}