#ifndef CNN_H
#define CNN_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

using namespace std;
using namespace cv;

class cnn
{
	private:
		double conv1Weight[16][1][5][5];
		double conv1Bias[16];
		double conv4Weight[32][16][5][5];
		double conv4Bias[32];
		double linear8Weight[256][800];
		double linear8Bias[256];
		double linear10Weight[43][256];
		double linear10Bias[43];

		double img[32][32];
		int classLabel;

	public:
		cnn();
		~cnn();

        bool getImg(Mat matImg);
		bool loadModel(string filename);
		int forward();
};


#endif
