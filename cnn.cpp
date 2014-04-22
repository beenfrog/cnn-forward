#include "cnn.h"
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;

cnn::cnn()
{

}

cnn::~cnn()
{

}

bool cnn::getImg(Mat matImg)
{
	Mat_<uchar>::iterator it = matImg.begin<uchar>();
	Mat_<uchar>::iterator end = matImg.end<uchar>();
	for(int i=0; it != end; ++it,++i)
	{
		*(&img[0][0]+i) = static_cast<double>(*it) / 255.0;
	}
	
	for(int i=0; i<32; ++i)
	{
		for(int j=0; j<32; ++j)
		{
			cout<<setprecision(12)<<img[i][j]<<" ";
		}
		cout<<endl;
	}

	Mat T(32,32,CV_64F,img);
	imshow("T",T);
	waitKey(9000);
}



bool cnn::loadModel(string filename)
{
	ifstream infile(filename.c_str());
	if(!infile.is_open())
	{
		return false;
	}

	//conv 1 weight 16*1*5*5
	for(int i=0; i<16; ++i)
	{
		for(int j=0; j<1; ++j)
		{
			for(int m=0; m<5; ++m)
			{
				for(int n=0; n<5; ++n)
				{
					infile>>conv1Weight[i][j][m][n];
				}
			}
		}
	}

	//conv 1 bias 16
	for(int i=0; i<16; ++i)
	{
		infile>>conv1Bias[i];
	}

	//conv 4 weight 32*16*5*5
	for(int i=0; i<32; ++i)
	{
		for(int j=0; j<16; ++j)
		{
			for(int m=0; m<5; ++m)
			{
				for(int n=0; n<5; ++n)
				{
					infile>>conv4Weight[i][j][m][n];
				}
			}
		}
	}

	//conv 4 bias 32
	for(int i=0; i<32; ++i)
	{
		infile>>conv4Bias[i];
	}

	//linear 8 weight 256*800
	for(int i=0; i<256; ++i)
	{
		for(int j=0; j<800; ++j)
		{
			infile>>linear8Weight[i][j];
		}
	}

	//linear 8 bias 256
	for(int i=0; i<256; ++i)
	{
		infile>>linear8Bias[i];
	}

	//linear 10 weight 43*256
	for(int i=0; i<43; ++i)
	{
		for(int j=0; j<256; ++j)
		{
			infile>>linear10Weight[i][j];
		}
	}

	//linear 10 bias 43
	for(int i=0; i<43; ++i)
	{
		infile>>linear10Bias[i];
	}

	//cout<<setprecision(16)<<linear10Bias[0]<<endl;
	return true;
}



int cnn::forward()
{
	return 1;
}
