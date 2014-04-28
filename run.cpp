#include "cnn.h"
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	// load the cnn model
	cnn trafficSign;
	trafficSign.loadModel("./data/gtsrb.txt");

	// forward the cnn
	stringstream ss;
	string filename;
	Mat img;
	for(int idx = 1; idx <= 99; ++idx)
	{
		ss.clear();
		ss<<"./data/png/"<<setw(5)<<setfill('0')<<idx<<".png";
		ss>>filename;
		img = imread(filename);

		trafficSign.getImg(img);//push the img to cnn
		cout<<trafficSign.forward()<<endl;//get the result
	}

	return 0;
}
