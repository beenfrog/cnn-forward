#include "cnn.h"
#include <sstream>
#include <string>
#include <iomanip>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	cnn trafficSign;
	trafficSign.loadModel("./data/gtsrb.txt");

	Mat matImg = imread("./data/png/00001.png");
	trafficSign.getImg(matImg);

/*
	stringstream ss;
	string filename;
	Mat _img;
	for(int idx = 1; idx <= 99; ++idx)
	{
		ss.clear();
		ss<<"./data/png/"<<setw(5)<<setfill('0')<<idx<<".png";
		ss>>filename;
		_img = imread(filename);

		trafficSign.getImg(_img);
		trafficSign.showImg();
	}
*/

	return 0;
}
