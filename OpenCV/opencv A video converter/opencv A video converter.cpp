/*********************************************************************************
making some changes in video
source: https://youtu.be/TJHlX8nhf40
result: https://www.youtube.com/watch?v=U0TL-5J0zrw&feature=youtu.be
*********************************************************************************/
#include <iostream>
#include <opencv2/opencv.hpp>    //opencv header
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	VideoCapture gcapture;

	string infilename;
	cout << "Input videol:";
	cin >> infilename;

	//try to open string, this will attempt to open it as a video file
	gcapture.open(infilename);
	if(!gcapture.isOpened()){
		cerr << "Failed to open a video device or cideo file!\n" << endl;
		return 1;
	}

	namedWindow("capture", WINDOW_AUTOSIZE);

	Size size = Size((int)gcapture.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)gcapture.get(CV_CAP_PROP_FRAME_HEIGHT));

	cout << size.width << "," << size.height << endl;
	cout << gcapture.get(CV_CAP_PROP_FPS) << endl;

	int totalFrame = gcapture.get(CV_CAP_PROP_FRAME_COUNT);

	//out file
	string outfilename;
	cout << "Output video:";
	cin >> outfilename;

	if(outfilename == ""){
		gcapture.release();
		return 1;
	}

	VideoWriter videoout(outfilename, CV_FOURCC('M', 'J', 'P', 'G'),
		gcapture.get(CV_CAP_PROP_FPS), size, true);

	if(!videoout.isOpened()){
		gcapture.release();
		return 1;
	}

	int key = 0;

	Mat colorFrame, tempFrame, frame;
	vector<Mat> sepFrame;

	for(int counter = 0;
		counter < totalFrame && tolower(key) != 'q' && key != 27;
		counter++){

#if 0 //you read a frame using operator >>
			gcapture >> frame;
			if(frame.emppt()) return 1;
#else // you may also read a frame using the member function read
			if(!gcapture.read(frame)) break;
			if(frame.empty()) break;
			cout << "frame in\n";
#endif
		//frame.copyTo(colorFrame);
		frame.convertTo(colorFrame, CV_32FC3);

		cvtColor(colorFrame, tempFrame, CV_BGR2HLS);

		split(tempFrame, sepFrame);

		Mat& hue = sepFrame[0];

		for(auto y = 0; y < hue.rows; y++){
			for(auto x = 0; x < hue.cols; x++){
				hue.at<float>(y,x) = 20;
			}
		}


		merge(sepFrame, colorFrame);
		cvtColor(colorFrame, tempFrame, CV_HLS2BGR);
		tempFrame.convertTo(colorFrame, CV_8UC3);
		//put my name
		putText(colorFrame,"00057155 CS 4B YISHAN LIN"
			,cvPoint(50,50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200,150,70), 1);

		//make several names
		if(rand()%100 > 70){
			for(int j = 51; j <60; j++)
				putText(colorFrame,"00057155 CS 4B YISHAN LIN"
					,cvPoint(j,j), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200,150,70), 1);			
		}


		//make lightning
		int rc = rand()%300;

		if(rand() % 100 > 90){
			for(int L = 0; L < 300+rc/20*20; L+=20){
				for(int k = 0; k < 20; k++){
					for(int j = 0; j < 5; j++){
						circle(colorFrame, Point(j+k+L*2,rc+k+100), 1, Scalar(255,255,255),-1);
						circle(colorFrame, Point(j+k+L*2,rc+k+100), 1, Scalar(255,255,255),-1);
						circle(colorFrame, Point(j+k+L*2,rc+k+100), 1, Scalar(255,255,255),-1);
						circle(colorFrame, Point(j+k+L*2,rc+k+100), 1, Scalar(255,255,255),-1);
						circle(colorFrame, Point(j+k+L*2,rc+k+100), 1, Scalar(255,255,255),-1);
						circle(colorFrame, Point(j+k+L*2,rc+k+100), 1, Scalar(255,255,255),-1);
					}
				}
				for(int k = 0; k < 20; k++){
					for(int j = 0; j < 5; j++){
						circle(colorFrame, Point(j+k+L*2+20,rc+120-k), 1, Scalar(255,255,255),-1);
						circle(colorFrame, Point(j+k+L*2+20,rc+120-k), 1, Scalar(255,255,255),-1);
						circle(colorFrame, Point(j+k+L*2+20,rc+120-k), 1, Scalar(255,255,255),-1);
						circle(colorFrame, Point(j+k+L*2+20,rc+120-k), 1, Scalar(255,255,255),-1);
						circle(colorFrame, Point(j+k+L*2+20,rc+120-k), 1, Scalar(255,255,255),-1);
						circle(colorFrame, Point(j+k+L*2+20,rc+120-k), 1, Scalar(255,255,255),-1);
					}
				}
			}
		}

		//make little circle
		rc = rand()%20000 + 5000;
		for(int k = 0; k < rc; k++){
			double ra = rand()%1000000 / 10;
			double rb = rand()%1000000 / 10;
			circle(colorFrame, Point(ra,rb), 1, Scalar(255,255,255),-1);
		}
		if(rand()%100 > 90){
			for(int i =1; i <25; i = i + 2){
					GaussianBlur(colorFrame, tempFrame, Size(i,i), 3, 3);
			}
		tempFrame.copyTo(colorFrame);
		}
		
		//show this frame on a window
		imshow("capture", colorFrame);

		//write this frame into the output video
		videoout << colorFrame;

		//you must call this function; otherwise,
		//you cannot see any images shown by imshow and 
		//cannot caputer any mouse events
		key = waitKey(20);

	}
	return 0;
}
