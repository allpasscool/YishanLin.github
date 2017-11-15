#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <omp.h>


const int INF = 10000000;
const int V = 10010;
void input(char *inFileName);
void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);

int n, m;	// Number of vertices, edges
int Dist[V][V];



__global__ void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int* Dist, int n)
{
	int block_end_x = block_start_x + block_height;
	int block_end_y = block_start_y + block_width;
	
	
		
	//for(int g = 0; g < n && threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0; g++){
	//			for(int a = 0; a < n; a++){
	//				if(Dist[g*n+a] == 10000000)
	//					printf("");
	//				else
	//					printf("%d ", Dist[g * n + a]);
	//				//printf("%d ", Dist[g][a]);
	//			}
	//			printf("\n");
	//			if(g == n-1)
	//				printf("############Dist_dev#######################\n");
	//}
	
	int b_i = blockIdx.x;
	int b_j = blockIdx.y;
	if(b_i < block_end_x && b_i >= block_start_x && (blockIdx.x * B + threadIdx.x) < n) {
		if(b_j < block_end_y && b_j >= block_start_y && (blockIdx.y * B + threadIdx.y) < n) {
			// To calculate B*B elements in the block (b_i, b_j)
			// For each block, it need to compute B times
			for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
				// To calculate original index of elements in the block (b_i, b_j)
				// For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
				//int block_internal_start_x = b_i * B;
				//int block_internal_end_x   = (b_i +1) * B;
				//int block_internal_start_y = b_j * B; 
				//int block_internal_end_y   = (b_j +1) * B;
				//if (block_internal_end_x > n)	block_internal_end_x = n;
				//if (block_internal_end_y > n)	block_internal_end_y = n;

				int i = blockIdx.x * B + threadIdx.x;
				int j = blockIdx.y * B + threadIdx.y;
				if(threadIdx.x < n && threadIdx.y < n && Dist[i*n+k] + Dist[k*n+j] < Dist[i*n+j] )
				{
					Dist[i*n+j] = Dist[i*n+k] + Dist[k*n+j];
				}
				
				
				__syncthreads();
				
				/*
				for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
					for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
						if (Dist[i][k] + Dist[k][j] < Dist[i][j])
							Dist[i][j] = Dist[i][k] + Dist[k][j];
					}
				}
				*/
			}
		}
	}
	__syncthreads();
		
	//for(int g = 0; g < n && threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0; g++){
	//			for(int a = 0; a < n; a++){
	//				printf("%d ", Dist[g * n + a]);
	//				//printf("%d ", Dist[g][a]);
	//			}
	//			printf("\n");
	//			if(g == n-1)
	//				printf("############Dist_dev before leaving kernel#######################\n");
	//}
}

__global__ void cal2(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int* Dist, int n)
{
	int block_end_x = block_start_x + block_height;
	int block_end_y = block_start_y + block_width;
	
	int b_i = blockIdx.x;
	int b_j = blockIdx.y;
	if(b_i < block_end_x && b_i >= block_start_x && (blockIdx.x * B + threadIdx.x) < n) {
		if(b_j < block_end_y && b_j >= block_start_y && (blockIdx.y * B + threadIdx.y) < n && (blockIdx.y *B + threadIdx.y) < (n/2)) {
			// To calculate B*B elements in the block (b_i, b_j)
			// For each block, it need to compute B times
			for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
				// To calculate original index of elements in the block (b_i, b_j)
				// For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2

				int i = blockIdx.x * B + threadIdx.x;
				int j = blockIdx.y * B + threadIdx.y;
				if(threadIdx.x < n && threadIdx.y < n && Dist[i*n+k] + Dist[k*n+j] < Dist[i*n+j] )
				{
					Dist[i*n+j] = Dist[i*n+k] + Dist[k*n+j];
				}
				
				
				__syncthreads();
				
			}
		}
	}
	__syncthreads();

}

__global__ void cal3(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int* Dist, int n)
{
	int block_end_x = block_start_x + block_height;
	int block_end_y = block_start_y + block_width;

	int b_i = blockIdx.x;
	int b_j = blockIdx.y;
	if(b_i < block_end_x && b_i >= block_start_x && (blockIdx.x * B + threadIdx.x) < n) {
		if(b_j < block_end_y && b_j >= block_start_y && (blockIdx.y * B + threadIdx.y) < n  && (blockIdx.y * B + threadIdx.y) >= (n/2)) {
			// To calculate B*B elements in the block (b_i, b_j)
			// For each block, it need to compute B times
			for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
				// To calculate original index of elements in the block (b_i, b_j)
				// For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2

				int i = blockIdx.x * B + threadIdx.x;
				int j = blockIdx.y * B + threadIdx.y;
				if(threadIdx.x < n && threadIdx.y < n && Dist[i*n+k] + Dist[k*n+j] < Dist[i*n+j] )
				{
					Dist[i*n+j] = Dist[i*n+k] + Dist[k*n+j];
				}
				
				
				__syncthreads();
			}
		}
	}
	__syncthreads();
}

void cal1(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height)
{
	int block_end_x = block_start_x + block_height;
	int block_end_y = block_start_y + block_width;

	for (int b_i =  block_start_x; b_i < block_end_x; ++b_i) {
		for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
			// To calculate B*B elements in the block (b_i, b_j)
			// For each block, it need to compute B times
			for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
				// To calculate original index of elements in the block (b_i, b_j)
				// For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
				int block_internal_start_x = b_i * B;
				int block_internal_end_x   = (b_i +1) * B;
				int block_internal_start_y = b_j * B; 
				int block_internal_end_y   = (b_j +1) * B;

				if (block_internal_end_x > n)	block_internal_end_x = n;
				if (block_internal_end_y > n)	block_internal_end_y = n;

				for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
					for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
						if (Dist[i][k] + Dist[k][j] < Dist[i][j])
							Dist[i][j] = Dist[i][k] + Dist[k][j];
					}
				}
			}
		}
	}
}


int main(int argc, char* argv[])
{
	struct timeval ts, tnow;
	gettimeofday(&ts, NULL);
	
	input(argv[1]);
	int B = atoi(argv[3]);
	if(B > 16){
		B = 16;
	}
	if(n <= B){
		B = n;
	}
	
	block_FW(B);

	output(argv[2]);
	
	gettimeofday(&tnow, NULL);
	printf("time: %d millisecond\n", ((tnow.tv_sec - ts.tv_sec) * 1000000 + (tnow.tv_usec - ts.tv_usec)) / 1000);

	return 0;
}

void input(char *inFileName)
{
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", &n, &m);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == j)	Dist[i][j] = 0;
			else		Dist[i][j] = INF;
		}
	}

	while (--m >= 0) {
		int a, b, v;
		fscanf(infile, "%d %d %d", &a, &b, &v);
		--a, --b;
		Dist[a][b] = v;
	}
}

void output(char *outFileName)
{
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (Dist[i][j] >= INF)	fprintf(outfile, "INF ");
			else					fprintf(outfile, "%d ", Dist[i][j]);
		}
		fprintf(outfile, "\n");
	}		
}

int ceil(int a, int b)
{
	return (a + b -1)/b;
}

void block_FW(int B)
{
	int *temp, *temp2;
	
	temp = (int*) malloc(n * n * sizeof(int));
	temp2 = (int*) malloc(n * n * sizeof(int));
	
	
	//cuda malloc
	int *Dist_dev, *Dist_dev2;
	cudaSetDevice(0);
	cudaMalloc((void **) &Dist_dev, n * n * sizeof(int));
	cudaSetDevice(1);
	cudaMalloc((void **) &Dist_dev2, n * n * sizeof(int));
	cudaSetDevice(0);
	
	//round 
	int round = ceil(n, B);
	
	//cuda block & threads
	dim3 threadsPerBlock(B, B);
	dim3 numOfBlock(round, round);
	
	//Dist to temp
	for(int a = 0, p = 0; a < n; a++){
		for(int b = 0; b < n; b++, p++){
			temp[p] = Dist[a][b];
		}
	}
	
	//for(int h = 0; h < n; h++){
	//	for(int g = 0; g < n; g++){
	//			if(temp[h*n+g] == 10000000)
	//				printf("");
	//			else
	//				printf("%d ", temp[h * n + g]);
	//	}
	//	printf("\n");
	//}
	//printf("------------------------------\n");
	omp_set_num_threads(2);
	
	for (int r = 0; r < round; ++r) {
	
		//cuda memory copy
		cudaMemcpy(Dist_dev, temp, n * n * sizeof(int), cudaMemcpyHostToDevice);
		//for(int h = 0; h < n; h++){
		//	for(int g = 0; g < n; g++){
		//		if(r <= 5)
		//			printf("%d ", temp[h * n + g]);
		//		//if(g == n-1)
		//			//printf("rrr %d\n", r);
		//	}
		//printf("\n");
		//}
		//printf("------------------------------\n");
		/* Phase 1*/
		cal <<<numOfBlock, threadsPerBlock>>> (B, r, r, r, 1, 1, Dist_dev, n);
		cudaMemcpy(temp, Dist_dev, n * n * sizeof(int), cudaMemcpyDeviceToHost);
		//temp to Dist
		for(int a = 0, p = 0; a < n; a++){
			for(int b = 0; b < n; b++, p++){
				Dist[a][b] = temp[p];
			}
		}
		//cal1(B, r, r, r, 1, 1);
		//printf("$$$$$$$$$$$$$$kernel finish$$$$$$$$$$$$$$$$$$$$$$$$\n");
		//for(int h = 0; h < n; h++){
		//	for(int g = 0; g < n; g++){
		//		if(r <= 5)
		//			printf("%d ", Dist[h][g]);
		//		//if(g == n-1)
		//			//printf("rrr %d\n", r);
		//	}
		//printf("\n");
		//}
		//printf("+++++++++++++++++ after phase1+++++++++++++++\n");
		
		
		/* Phase 2*/
		//cudaMalloc((void **) &Dist_dev, n * n * sizeof(int));
		cudaMemcpy(Dist_dev, temp, n * n * sizeof(int), cudaMemcpyHostToDevice);
		cal <<<numOfBlock, threadsPerBlock>>> (B, r, 	r, 		0, 		r, 			1, Dist_dev, n);		
		cudaMemcpy(temp, Dist_dev, n * n * sizeof(int), cudaMemcpyDeviceToHost);
		//temp to Dist
		for(int a = 0, p = 0; a < n; a++){
			for(int b = 0; b < n; b++, p++){
				Dist[a][b] = temp[p];
			}
		}
		
		cal <<<numOfBlock, threadsPerBlock>>> (B, r, 	r,		r+1, 	round-r-1, 	1, Dist_dev, n);
		cal <<<numOfBlock, threadsPerBlock>>> (B, r, 	0, 		r, 		1, 			r, Dist_dev, n);
		
		//cudaMalloc((void **) &Dist_dev, n * n * sizeof(int));
		//cudaMemcpy(Dist_dev, temp, n * n * sizeof(int), cudaMemcpyHostToDevice);
		//cal <<<numOfBlock, threadsPerBlock>>> (B, r, 	r,		r+1, 	round-r-1, 	1, Dist_dev, n);
		//cudaMemcpy(temp, Dist_dev, n * n * sizeof(int), cudaMemcpyDeviceToHost);
		//cudaMalloc((void **) &Dist_dev, n * n * sizeof(int));
		//cudaMemcpy(Dist_dev, temp, n * n * sizeof(int), cudaMemcpyHostToDevice);
		//cal <<<numOfBlock, threadsPerBlock>>> (B, r, 	0, 		r, 		1, 			r, Dist_dev, n);
		//cudaMemcpy(temp, Dist_dev, n * n * sizeof(int), cudaMemcpyDeviceToHost);
		//cudaMalloc((void **) &Dist_dev, n * n * sizeof(int));
		//cudaMemcpy(Dist_dev, temp, n * n * sizeof(int), cudaMemcpyHostToDevice);
		cal <<<numOfBlock, threadsPerBlock>>> (B, r, 	r+1, 	r, 		1, 			round-r-1, Dist_dev, n);
		cudaMemcpy(temp, Dist_dev, n * n * sizeof(int), cudaMemcpyDeviceToHost);
		//cal1(B, r,     r,     0,             r,             1);
		//cal1(B, r,     r,  r +1,  round - r -1,             1);
		//cal1(B, r,     0,     r,             1,             r);
		//cal1(B, r,  r +1,     r,             1,  round - r -1);
		//temp to Dist
		for(int a = 0, p = 0; a < n; a++){
			for(int b = 0; b < n; b++, p++){
				Dist[a][b] = temp[p];
			}
		}
		//for(int h = 0; h < n; h++){
		//	for(int g = 0; g < n; g++){
		//		if(r <= 5)
		//			printf("%d ", Dist[h][g]);
		//		//if(g == n-1)
		//			//printf("rrr %d\n", r);
		//	}
		//printf("\n");
		//}
		//printf("+++++++++++++++++ after phase2+++++++++++++++\n");
		
		
		/* Phase 3*/
		cudaMemcpy(Dist_dev, temp, n * n * sizeof(int), cudaMemcpyHostToDevice);
		cal <<<numOfBlock, threadsPerBlock>>> (B, r, 	0, 		0, 		r, 			r, 			Dist_dev, n);
		cudaMemcpy(temp, Dist_dev, n * n * sizeof(int), cudaMemcpyDeviceToHost);
		
		//cudaMemcpy(Dist_dev, temp, n * n * sizeof(int), cudaMemcpyHostToDevice);
		
		#pragma omp parallel
		{
			int cpu_thread_id = omp_get_thread_num();
			//printf("%d %d %d %d\n", cpu_thread_id, temp[0], temp[1], temp[2]);
			if(cpu_thread_id == 0){
				cudaSetDevice(0);
				cudaMemcpy(Dist_dev, temp, n * n * sizeof(int), cudaMemcpyHostToDevice);
				cal2 <<<numOfBlock, threadsPerBlock>>> (B, r, 	0, 		r+1,	round-r-1, 	r, 			Dist_dev, n);
				cudaMemcpy(temp, Dist_dev, n * n * sizeof(int), cudaMemcpyDeviceToHost);
			}
			else if(cpu_thread_id == 1){
				cudaSetDevice(1);
				cudaMemcpy(Dist_dev2, temp, n * n * sizeof(int), cudaMemcpyHostToDevice);
				cal3 <<<numOfBlock, threadsPerBlock>>> (B, r, 	0, 		r+1,	round-r-1, 	r, 			Dist_dev2, n);
				cudaMemcpy(temp2, Dist_dev2, n * n * sizeof(int), cudaMemcpyDeviceToHost);
			}
			//printf("%d %d %d %d\n", cpu_thread_id, temp[0], temp[1], temp[2]);				
		}
		cudaSetDevice(0);
		/*printf("0\n");
		for(int h = 0; h < n; h++){
			for(int g = 0; g < n; g++){
				if(r <= 5)
					printf("%d ", temp[h*n+g]);
			}
			printf("\n");
		}
		printf("1\n");
		for(int h = 0; h < n; h++){
			for(int g = 0; g < n; g++){
				if(r <= 5)
					printf("%d ", temp2[h*n+g]);
			}
			printf("\n");
		}*/
		for(int f = 0; f < n*n; f++){
			if(temp2[f] < temp[f])
				temp[f] = temp2[f];
		}
		//cudaMemcpy(temp, Dist_dev, n * n * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(Dist_dev, temp, n * n * sizeof(int), cudaMemcpyHostToDevice);
		cal <<<numOfBlock, threadsPerBlock>>> (B, r, 	r+1,	0, 		r, 			round-r-1,	Dist_dev, n);
		//cudaMemcpy(temp, Dist_dev, n * n * sizeof(int), cudaMemcpyDeviceToHost);
		//cudaMemcpy(Dist_dev, temp, n * n * sizeof(int), cudaMemcpyHostToDevice);
		cal <<<numOfBlock, threadsPerBlock>>> (B, r, 	r+1, 	r+1, 	round-r-1, 	round-r-1, 	Dist_dev, n);
		//printf("OK\n");
		cudaMemcpy(temp, Dist_dev, n * n * sizeof(int), cudaMemcpyDeviceToHost);
		//cal1(B, r,     0,     0,            r,             r);
		//cal1(B, r,     0,  r +1,  round -r -1,             r);
		//cal1(B, r,  r +1,     0,            r,  round - r -1);
		//cal1(B, r,  r +1,  r +1,  round -r -1,  round - r -1);
		//get the answer back
		
		
		//Dist
		//for(int h = 0; h < n; h++){
		//	for(int g = 0; g < n; g++){
		//		if(r <= 5)
		//			printf("%d ", Dist[h][g]);
		//		//if(g == n-1)
		//			//printf("rrr %d\n", r);
		//	}
		//printf("\n");
		//}
		//printf("+++++++++++++++++Dist after phase2 , after phase3 before temp to Dist+++++++++++++++\n");
		
		
		
		//temp to Dist
		for(int a = 0, p = 0; a < n; a++){
			for(int b = 0; b < n; b++, p++){
				Dist[a][b] = temp[p];
			}
		}
		//for(int h = 0; h < n; h++){
		//	for(int g = 0; g < n; g++){
		//		if(temp[h*n+g] == 10000000)
		//			printf("");
		//		else
		//			printf("%d ", temp[h*n + g]);
		//		//if(g == n-1)
		//			//printf("rrr %d\n", r);
		//	}
		//	printf("\n");
		//}
		//printf("+++++++++++++++++temp after phase3+++++++++++++++\n");
		
		
		
		//for(int h = 0; h < n; h++){
		//	for(int g = 0; g < n; g++){
		//		if(r <= 5)
		//			printf("%d ", Dist[h][g]);
		//		//if(g == n-1)
		//			//printf("rrr %d\n", r);
		//	}
		//printf("\n");
		//}
		//printf("+++++++++++++++++Dist after phase3+++++++++++++++\n");
	}
	
	cudaFree(Dist_dev);
}