/***********************************************************************
result:https://www.youtube.com/watch?v=blOu_IV9Dws&feature=youtu.be
An augmented reality game.
*********************************************************************/

#include<iostream>
#include<string>
#include<math.h>
#include<time.h>
#include<sstream>
#include<climits>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace std;
using namespace cv;

//--------------------------------------------------------------------------------------
void skin_color(Mat& color,Mat_<uchar>& mask,int R_thr=95,int G_thr=40,int B_thr=20,int Max_min_diff_thr=15,int R_G_diff_thr=15) 
{
    // color¬°BGR¼v¹³
	for(int y = 0;y < color.rows; ++y) {
		Vec3b* b = (Vec3b*)color.ptr(y);
		Vec3b* eb= b+color.cols;
		uchar*p = mask.ptr(y);
		memset(p,0,mask.cols);
		for(;b != eb; ++b,++p) {
			Vec3b& px = *b;
			if (px[2]>R_thr&& px[0]>B_thr && px[1]>G_thr &&
				px[2]>px[1] && px[2]>px[0] && (px[2]-px[1])>R_G_diff_thr&&   
				(px[2]-(px[0]<=px[1]?px[0]:px[1]))>Max_min_diff_thr) {
				*p = 1;
			}
		}
	}
	return;
}
//--------------------------------------------------------------------------------------
void show_pic(Mat& frame,pair<Mat,Mat>& pic,int x,int y)
{	
	Mat canvas = Mat(frame,Rect(x,y,pic.first.cols,pic.first.rows));
	cvAnd(&((CvMat)canvas),&((CvMat)pic.second),&((CvMat)canvas));
	cvOr(&((CvMat)canvas),&((CvMat)pic.first),&((CvMat)canvas));
}

//--------------------------------------------------------------------------------------
int main(void)
{

	VideoCapture gcapture;

	gcapture.open(0);// (arg); //try to open string, this will attempt to open it as a video file

	if (!gcapture.isOpened()) {
		cerr << "Failed to open a video device or video file!\n" << endl;
		return 1;
	}

	srand(time(NULL));

	namedWindow("capture",WINDOW_AUTOSIZE);

	// The frame size of the output video
	Size size=Size((int)gcapture.get(CV_CAP_PROP_FRAME_WIDTH)*2,(int) gcapture.get(CV_CAP_PROP_FRAME_HEIGHT));

	cout << size.width << "," << size.height << endl;
	cout << gcapture.get(CV_CAP_PROP_FPS) << endl;

	int showBomb;
	int bombPos;
	bool pk1 = true;
	bool pk2 = true;
	bool pk3 = true;
	int temp = 0;
	int firstFrame = true;
	string outfilename = "hw3.avi";
	pair<Mat,Mat> ball;
	pair<Mat,Mat> cannonorg;
	pair<Mat,Mat> bb;
	pair<Mat,Mat> cannon;
	pair<Mat, Mat> pika;
	pair<Mat, Mat> pika1;
	pair<Mat, Mat> pika2;
	pair<Mat, Mat> bomb;
	showBomb = 0;
	bombPos = size.height;
	ball.first      = imread("ball.png",1);
	ball.second     = imread("ball_mask.png",1);
	cannonorg.first = imread("cannon2.png",1);
	cannonorg.second= imread("cannon_mask2.png",1);
	bb.first        = imread("bb.png",1);
	bb.second       = imread("bb_mask.png",1);
	pika.first      = imread("pikachu.png",1);
	pika.second		= imread("pikachu_mask.png",1);
	pika1.first     = imread("pikachu1.png",1);
	pika1.second	= imread("pikachu_mask1.png",1);
	pika2.first     = imread("pikachu2.png",1);
	pika2.second	= imread("pikachu_mask2.png",1);
	bomb.first		= imread("bomb.png",1);
	bomb.second		= imread("bomb_mask.png",1);

	int key = 0;

	int initX = cannonorg.first.cols/2;
	int initY = gcapture.get(CV_CAP_PROP_FRAME_HEIGHT)-cannonorg.first.rows-1;

	Mat canvas(gcapture.get(CV_CAP_PROP_FRAME_HEIGHT),2*gcapture.get(CV_CAP_PROP_FRAME_WIDTH),CV_8UC3);
	Mat cmask = Mat(canvas,Rect(0,0,gcapture.get(CV_CAP_PROP_FRAME_WIDTH),gcapture.get(CV_CAP_PROP_FRAME_HEIGHT)));
	Mat frame = Mat(canvas,Rect(gcapture.get(CV_CAP_PROP_FRAME_WIDTH),0,gcapture.get(CV_CAP_PROP_FRAME_WIDTH),gcapture.get(CV_CAP_PROP_FRAME_HEIGHT)));
	Mat_<uchar> mask(gcapture.get(CV_CAP_PROP_FRAME_HEIGHT),gcapture.get(CV_CAP_PROP_FRAME_WIDTH));
	Mat curframe;
	Mat cframe;
	vector<int> hskin(gcapture.get(CV_CAP_PROP_FRAME_WIDTH));

	VideoWriter videoout(outfilename, CV_FOURCC('M', 'J', 'P', 'G'), gcapture.get(CV_CAP_PROP_FPS), size, true);

	Point ballPos;
	
	ballPos.x = 1000;
	ballPos.y = initY;

	Point2f speed;	

	float G = 4;

	int roi_x0 = 0;
	int roi_x1 = gcapture.get(CV_CAP_PROP_FRAME_WIDTH)-ball.first.cols-10;
	int roi_y0 = 0;
	int roi_y1 = gcapture.get(CV_CAP_PROP_FRAME_HEIGHT)-ball.first.rows-10;

	int countdown;
	int score = 0;
	int bbPos = 0;
	bool catchball;
	int totalB = 0;

	int startT = clock();

	for(;tolower(key) != 'q' && key != 27;) {		
		
		if (ballPos.x >= roi_x1||ballPos.y>=roi_y1) {
			// fire a ball
			float cannon_angle = -(rand()%25+60.)/180.*CV_PI; // cannon angle
			float s            = rand()%20+35; // initial ball speed

			float cs           = cos(cannon_angle);
			float sn           = sin(cannon_angle);

			speed.x            = s * cs;
			speed.y            = s * sn;

			Mat R = (Mat_<float>(2,3) << cs, -sn, (1-cs)*cannonorg.first.cols/2+sn*cannonorg.first.rows/2, 
				                         sn, cs, -sn*cannonorg.first.cols/2+(1-cs)*cannonorg.first.rows/2);

			warpAffine(cannonorg.first,cannon.first,R,Size(cannonorg.first.cols,cannonorg.first.rows));
			warpAffine(cannonorg.second,cannon.second,R,Size(cannonorg.first.cols,cannonorg.first.rows),1,0,Scalar(255,255,255));

			float r     = 50;

			ballPos.x   = cannon.first.cols/2-cs*r;
			ballPos.y   = gcapture.get(CV_CAP_PROP_FRAME_HEIGHT)-cannon.first.rows/2+sn*r;
			bombPos = ballPos.y;

			catchball   = false;
			countdown   = 20;
			totalB++;  
		}

		gcapture >> curframe;

		if (curframe.empty()) return 1;

		curframe.copyTo(cframe); 

		flip(cframe,cframe,1);   // flip the image horizontally

		cframe.copyTo(frame);

		skin_color(cframe,mask); // detect pixels of skin color

		int maxid = 0;
		int maxv  = INT_MIN;
		for(int i = 0; i < mask.cols; ++i) {
			hskin[i] = sum(Mat(mask,Rect(i,0,1,mask.rows)))[0];
			int temp;
			if (hskin[i] > maxv) {
				maxid = i;
				maxv  = hskin[i];
			}
		}

		int newbbPos = maxid - bb.first.cols/2;
		if (newbbPos < 0) {
			newbbPos = 0;
		} else if (newbbPos+bb.first.cols >= mask.cols) {
			newbbPos = mask.cols-bb.first.cols-1;
		}

		if (abs(newbbPos-bbPos)>=15) bbPos = newbbPos;

		show_pic(frame,bb,bbPos,frame.rows-bb.first.rows-1);
		if(pk1)
			show_pic(frame,pika, 100, 0);
		if(pk2)
			show_pic(frame,pika1, 250, 0);
		if(pk3)	
			show_pic(frame,pika2, 400, 0);	
		
		if (countdown == 0) {

			if(catchball==false&&ballPos.y>=frame.rows-bb.first.rows&&
			   ballPos.x+ball.first.cols/2>=bbPos&&ballPos.x+ball.first.cols/2<bbPos+bb.first.cols) {
				catchball = true;
				score++;
				showBomb = 1;
			}
			if (ballPos.x >= roi_x0 && ballPos.x < roi_x1 && ballPos.y >= roi_y0 && ballPos.y < roi_y1) {
				show_pic(frame,ball,ballPos.x,ballPos.y);
			}


			ballPos.x  += speed.x;
			ballPos.y  += speed.y;
			speed.y    += G;

			if(showBomb && bombPos != 0){
				if(showBomb == 2){
					for(int i = 15; i != 0; i--){
						show_pic(frame, bomb, bbPos, bombPos);
						bombPos--;
						if(bombPos == 0){
							showBomb = 0;
							break;
						}
						if(bombPos == 10){
							showBomb = 0;
							break;
						}
					}
				}
				else{showBomb++;}
				if(bbPos <= 200 && bbPos >= 60 && bombPos <= 150){
					if(temp == 1){
						pk1 = false;
						temp = 0;
						showBomb = 0;
					}
					else
						temp++;
				}
				if(bbPos <= 320 && bbPos >= 220 && bombPos <= 150)
					if(temp == 1){
						pk2 = false;
						temp = 0;
						showBomb = 0;
					}
					else
						temp++;
				if(bbPos <= 470 && bbPos >= 330 && bombPos <= 150)
					if(temp == 1){
						pk3 = false;
						temp = 0;
						showBomb = 0;
					}
					else
						temp++;
		}

		} else {
			countdown--;
		}

		ostringstream scorestr,timestr,totalballstr;
		scorestr << score;
		timestr<<(clock()-startT)/CLOCKS_PER_SEC;
		totalballstr<<totalB;
		putText(frame,string("Score:")+scorestr.str()+string("/")+totalballstr.str()+string(" Time:")+timestr.str(),
			    Point(300,25),FONT_HERSHEY_SIMPLEX,0.8,Scalar(0,0,255),2);

		show_pic(frame,cannon,0,frame.rows-cannon.first.rows-1);

		mask*=255;
		cvtColor(mask,cmask,CV_GRAY2BGR);

		for(int i = 1; i < hskin.size(); ++i) {
			line(cmask,Point(i-1,cmask.rows-hskin[i-1]),Point(i,cmask.rows-hskin[i]),Scalar(0,0,255));
		}

		imshow("capture", canvas);
		if (firstFrame) {
			videoout.open("hw3.avi",CV_FOURCC('M','J','P','G'),30,canvas.size(),true);
			firstFrame = false;
		}
		videoout << canvas;

		key = waitKey(20);
		if(!pk1 && !pk2 && !pk3)
			break;
	}

	return 0;
}