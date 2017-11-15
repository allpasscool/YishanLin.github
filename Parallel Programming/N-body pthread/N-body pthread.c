#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <X11/Xlib.h>    // for X-window 
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <math.h>
#include <string.h>

double const g = 0.0000000000667384;

//body struct
struct Body{
	double x, y, newx, newy;
	double vx, vy, newvx, newvy;
	double w;
	double t;
	int bodiesNum;
};

//calculate 
struct BodyCal{
	struct Body *Nbody;
	int Iam;
};

//calculate
void *calculateNewInfo(void *send){
	int i, j;
	struct BodyCal *thisBody;
	thisBody = (struct BodyCal*) send;
	i = thisBody[0].Iam;
	//calculate
	for(j = 0; j < thisBody[0].Nbody[0].bodiesNum; j++){
		if(j == i){
			continue;
		}
		double delta_x = thisBody[0].Nbody[i].x - thisBody[0].Nbody[j].x;
		double delta_y = thisBody[0].Nbody[i].y - thisBody[0].Nbody[j].y;
		double distance = sqrt(delta_x * delta_x + delta_y * delta_y);
		if(distance < 0.001){
			distance += 1.001;
		}
		double F = g * thisBody[0].Nbody[j].w / ((distance) * (distance));
		thisBody[0].Nbody[i].newvx = thisBody[0].Nbody[i].newvx - F * thisBody[0].Nbody[i].t * delta_x / distance;
		thisBody[0].Nbody[i].newvy = thisBody[0].Nbody[i].newvy - F * thisBody[0].Nbody[i].t * delta_y / distance;
		
	}
	thisBody[0].Nbody[i].newx = thisBody[0].Nbody[i].newx + thisBody[0].Nbody[i].newvx * thisBody[0].Nbody[i].t;
	thisBody[0].Nbody[i].newy = thisBody[0].Nbody[i].newy + thisBody[0].Nbody[i].newvy * thisBody[0].Nbody[i].t;

	pthread_exit(NULL);
}

//update
void *updateNewInfo(void *send){
	int i, j;
	struct BodyCal *thisBody;
	thisBody = (struct BodyCal*) send;
	i = thisBody[0].Iam;
	
	//update
	thisBody[0].Nbody[i].x = thisBody[0].Nbody[i].newx;
	thisBody[0].Nbody[i].y = thisBody[0].Nbody[i].newy;
	thisBody[0].Nbody[i].vx = thisBody[0].Nbody[i].newvx;
	thisBody[0].Nbody[i].vy = thisBody[0].Nbody[i].newvy;
	
	pthread_exit(NULL);
}


int main(int argc, char *argv[]){
	
	//declare variable
	struct timeval ts, tnow;
	unsigned  long diff;
	int threadSize, bodies, i, j, k, p, T, ss, snow;
	gettimeofday(&ts, NULL);
	ss = (int) (ts.tv_usec) /1000;
	double m, t, theta, xmin, ymin, length, Length;
	char* in_file = argv[5];//FILE: input file name
	char* enable = argv[7];//enable/disable: enable/disable Xwindow
	threadSize = atoi(argv[1]);//#threads: number of threads
	m = atof(argv[2]);//m: mass, is double precision floating point number
	T = atoi(argv[3]);//T: number of steps
	t = atof(argv[4]);//t: time between each step, is double precision floating point number
	theta = atof(argv[6]);//use in Barnes-Hut algorithm
	if(strcmp(enable, "enable") == 0){
		xmin = atof(argv[8]);//xmin: the upper left coordinate of Xwindow
		ymin = atof(argv[9]);//ymin: the upper left coordinate of Xwindow
		length = atof(argv[10]);//length: the length of coordinate axis
		Length = atof(argv[11]);//Length: the length of Xwindow’s side
	}
	struct Body *Nbody;
	struct BodyCal *NbodyCal;
	pthread_t *threads;
	pthread_attr_t attr;
	FILE *fpIn = fopen(in_file, "r");
	Window win;
	GC gc;
	Display *display;
	
	if(strcmp(enable, "enable") == 0){
		//printf("98\n");
		display = XOpenDisplay(NULL);if(display == NULL) printf("No Xwin\n");
		win = XCreateSimpleWindow(display, DefaultRootWindow(display), xmin, ymin, Length, Length, 1, WhitePixel( display, DefaultScreen( display ) ), BlackPixel( display, DefaultScreen( display ) ));
		XMapWindow( display, win );
		gc = XCreateGC( display, win, 0, 0 );
		XSetForeground( display, gc, WhitePixel( display, DefaultScreen( display ) ));
	}
	
	//malloc threads
	threads = (pthread_t*) malloc(threadSize * sizeof(pthread_t));
	
	//thread
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	
	//read Nbody
	fscanf(fpIn, "%d", &bodies);
	Nbody = (struct Body*) malloc(bodies * sizeof(struct Body));
	for(i = 0; i < bodies; i++){
		fscanf(fpIn, "%lf %lf %lf %lf", &(Nbody[i].x), &(Nbody[i].y), &(Nbody[i].vx), &(Nbody[i].vy));
		Nbody[i].w = m;
		Nbody[i].t = t;
		Nbody[i].bodiesNum = bodies;
		Nbody[i].newvx = Nbody[i].vx;
		Nbody[i].newvy = Nbody[i].vy;
		Nbody[i].newx = Nbody[i].x;
		Nbody[i].newy = Nbody[i].y;
	}
	
	//N body for calculating
	NbodyCal = (struct BodyCal*) malloc(bodies * sizeof(struct BodyCal));
	for(i = 0; i < bodies; i++){
		NbodyCal[i].Nbody = Nbody;
		NbodyCal[i].Iam = i;
	}
	
	//calculate
	//T times of steps
	for(i = 0; i < T; i++){
		//calculate
		
		for(j = 0;;j++){
			for(k = 0; k < threadSize; k++){
				if(j * threadSize + k == bodies){
					break;
				}
				
				pthread_create(&threads[k], NULL, calculateNewInfo, (void*) &(NbodyCal[j * threadSize + k]));
			}
			//join
			for(p = 0; p < k; p++){
				pthread_join(threads[p], NULL);
			}	
			if(j * threadSize + k == bodies){
					break;
			}
		}
		
		//update
		for(j = 0;;j++){
			for(k = 0; k < threadSize; k++){
				if(j * threadSize + k == bodies){
					break;
				}
				
				pthread_create(&threads[k], NULL, updateNewInfo, (void*)&(NbodyCal[j * threadSize + k]));
			}
			//join
			for(p = 0; p < k; p++){
				pthread_join(threads[p], NULL);
			}	
			if(j * threadSize + k == bodies){
					break;
			}
		}
		
		//draw and display
		if(strcmp(enable, "enable") == 0){
			for(p = 0; p < bodies; p++){
				XSetForeground(display,gc,WhitePixel(display,DefaultScreen( display )));
				XDrawPoint(display, win, gc, (int)((Nbody[p].x -xmin) * (Length / length)), (int)((Nbody[p].y - ymin) * (Length / length)));
			}
			XFlush(display);
			
			for(p = 0; p < bodies; p++){
				XSetForeground(display,gc,BlackPixel(display,DefaultScreen( display )));
				XDrawPoint(display, win, gc, (int)((Nbody[p].x -xmin) * (Length / length)), (int)((Nbody[p].y - ymin) * (Length / length)));
			}
			
		}
	}
	gettimeofday(&tnow, NULL);
	snow = (int)(tnow.tv_usec) /1000;
	diff = ((tnow.tv_sec - ts.tv_sec) * 1000000 + (tnow.tv_usec - ts.tv_usec)) / 1000;
	printf("total %ld millisec.\n", diff);
	pthread_attr_destroy(&attr);
	pthread_exit(NULL);
	
	//close file
	fclose(fpIn);
	
	return 0;
}