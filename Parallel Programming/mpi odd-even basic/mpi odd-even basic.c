//Homework1 s104062548

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

int main (int argc, char *argv[]) {
        //declare variables
		int rank, size, rc, i, start, end, *chunk, mysize, totalSize, k, ierr;
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

		//printf("34 OK SIZE:%d  N:%d\n", size, atoi(argv[1]));
		
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
	
		//declare list 
		//int *root_arr; // for root process (which rank == 0) only

        // Part 1: Read file
        /* Note: You should deal with cases where (N < size) in Homework 1 */
		MPI_File in, out;//printf("57 OK\nin:%s\nout:%s\n", inName,outName);
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
		totalSize = size;//printf("60 OK\n");
		//N > =size
		if(N >= size){
			//initialize
			start = N / size *rank;//printf("64 OK\n");
			if(rank != size-1){
				end = N / size * (rank + 1) - 1;
			}
			else{
				end = N - 1;
			}
			mysize = end - start + 1;//printf("71 OK\n");
			chunk = malloc((end - start + 1 + 1) * sizeof(MPI_INT));
			Ttemp = MPI_Wtime();
            MPI_File_read_at_all(in, (MPI_Offset)(start*sizeof(int)), chunk, mysize, MPI_INT, MPI_STATUS_IGNORE);
			TIO += MPI_Wtime() - Ttemp;
			//printf("read:%d  RANK:%d\n", chunk[0], rank);
		}
		//N < size
        else if(N < size) {
                start = rank;//printf("78 OK RANK:%d Start:%d\n", rank, start);
				if(rank < N){
					mysize = 1;
				}
				else{
					mysize = 0;
				} 
				chunk = malloc(2* sizeof(int));
				end = rank;
				//printf("92 OK RANK:%d Start:%d\n", rank, start);
				Ttemp = MPI_Wtime();
				MPI_File_read_at_all(in, (MPI_Offset)(start*sizeof(int)), chunk, mysize, MPI_INT, MPI_STATUS_IGNORE);
				TIO += MPI_Wtime() - Ttemp;
				size = N;//printf("85 OK RANK:%d\n", rank);
        }
		Ttemp = MPI_Wtime();
		MPI_File_close(&in);
		TIO += MPI_Wtime() - Ttemp;
		//printf("87OK RANK:%d\n", rank);
		//Part 2: odd-even sort
		int sorted = 0;
		if(rank < size){
			while(!sorted){
				sorted = 1;
				//odd sort
				i = 0;
				if(start %2 == 0){
					i++;
				}//printf("96OK\n");
				for(; i + start <= end; i+=2){//printf("97OK\n");
					if(i == 0){
						Ttemp = MPI_Wtime();
						MPI_Send(&chunk[i], 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
						MPI_Recv(&chunk[i], 1, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
						TComm += MPI_Wtime() - Ttemp;
					}
					else{
						if(chunk[i] < chunk[i-1]){
							sorted = 0;
							swap(&chunk[i], &chunk[i-1]);
						}
					}
				}//printf("108OK\n");
				if(rank != size -1 && end % 2 == 0){
					Ttemp = MPI_Wtime();
					MPI_Recv(&k, 1, MPI_INT, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
					if(k < chunk[end - start]){
						sorted = 0;
						swap(&k, &chunk[end - start]);
					}
					MPI_Send(&k, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
					TComm += MPI_Wtime() - Ttemp;
				}
				//even sort
				i = 0;//printf("118OK\n");
				if(rank == 0){
					i+=2;
				}
				if(start %2 ==1){
					i++;
				}
				for(; i + start <= end; i+=2){//printf("125OK\n");
					if(i == 0){
						Ttemp = MPI_Wtime();
						MPI_Send(&chunk[i], 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
						MPI_Recv(&chunk[i], 1, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
						TComm += MPI_Wtime() - Ttemp;
					}
					else{
						if(chunk[i] < chunk[i-1]){
							sorted = 0;
							swap(&chunk[i], &chunk[i-1]);
						}
					}
				}
				if(rank != size-1 && end % 2 == 1){
					Ttemp = MPI_Wtime();
					MPI_Recv(&k, 1, MPI_INT, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
					if(k < chunk[end - start]){
						sorted = 0;
						swap(&k, &chunk[end - start]);
					}
					MPI_Send(&k, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
					TComm += MPI_Wtime() - Ttemp;
				}//printf("144OK, RANK:%d\n", rank);
				//sorted
				if(size >= 2){
					if(rank == 0){//printf("166OK, RANK:%d\n", rank);
						Ttemp = MPI_Wtime();
						MPI_Send(&sorted, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
						MPI_Recv(&k, 1, MPI_INT, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
						TComm += MPI_Wtime() - Ttemp;
						if(k == 0) sorted = k;
					}
					else if(rank == size - 1){//printf("171OK, RANK:%d\n", rank);
						Ttemp = MPI_Wtime();
						MPI_Recv(&k, 1, MPI_INT, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
						if(k == 0) sorted = k;
						MPI_Send(&sorted, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
						TComm += MPI_Wtime() - Ttemp;
					}
					else{//printf("176OK, RANK:%d\n", rank);
						Ttemp = MPI_Wtime();
						MPI_Recv(&k, 1, MPI_INT, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
						if(k == 0) sorted = k;
						MPI_Send(&sorted, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
						MPI_Recv(&k, 1, MPI_INT, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
						if(k == 0) sorted = k;
						MPI_Send(&sorted, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
						TComm += MPI_Wtime() - Ttemp;
					}//printf("183OK, RANK:%d\n", rank);
				}//printf("184OK, RANK:%d\n", rank);
			}
			//printf("186OK\n");
		}
		// write file
		Ttemp = MPI_Wtime();
		MPI_File_write_at_all(out, (MPI_Offset)(start*sizeof(int)), chunk, mysize, MPI_INT, MPI_STATUS_IGNORE);
		TIO += MPI_Wtime() - Ttemp;
		//printf("190OK\nstart:%d chunk[0]:%d mysize:%d\n", start, chunk[0], mysize);
		free(chunk);
		//printf("185OK  RANK:%d", rank);
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
			/*else{
				printf("rank:%d  sorted:%d", rank, sorted);
			}*/
		}
		//printf("187OK\n");
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