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
		double conv1tanh2[16][28][28];
		double pooling3[16][14][14];
		double conv4tanh5[32][10][10];
		double pooling6[32][5][5];
		double reshape7[800];
		double linear8tanh9[256];
		double linear10[43];
		int classLabel;

		void forward12();
		void forward3();
		void forward45();
		void forward6();
		void forward7();
		void forward89();
		void forward10();

	public:
		cnn();
		~cnn();

		bool loadModel(string filename);
        bool getImg(Mat matImg);
		int forward();
};


#endif
