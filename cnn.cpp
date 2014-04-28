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
	if(matImg.empty())
	{
		return false;
	}
	else
	{
		Mat_<uchar>::iterator it = matImg.begin<uchar>();
		Mat_<uchar>::iterator end = matImg.end<uchar>();
		for(int i=0; it != end; ++it,++i)
		{
			*(&img[0][0]+i) = static_cast<double>(*it) / 255.0;
		}

		return true;
	}
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

//conv1 and tanh2
void cnn::forward12()
{
	for(int t=0; t<16; ++t)//layer to
	{
		for(int i=0; i<28; ++i)//to image row
		{
			for(int j=0; j<28; ++j)//to image col
			{
				conv1tanh2[t][i][j] = 0;
				for(int f=0; f<1; ++f)//layer from
				{
					for(int m=0; m<5; ++m)//filter row
					{
						for(int n=0; n<5; ++n)//filter col
						{
							conv1tanh2[t][i][j] += img[i+m][j+n] * conv1Weight[t][f][m][n];
						}
					}
				}
				conv1tanh2[t][i][j] += conv1Bias[t];
				conv1tanh2[t][i][j] = tanh(conv1tanh2[t][i][j]);
			}
		}
	}
}

//pooling3
void cnn::forward3()
{
	for(int t=0; t<16; ++t)
	{
		for(int i=0; i<14; ++i)
		{
			for(int j=0; j<14; ++j)
			{
				pooling3[t][i][j] = 0;
				for(int m=0; m<2; ++m)
				{
					for(int n=0; n<2; ++n)
					{
						pooling3[t][i][j] += pow(conv1tanh2[t][2*i+m][2*j+n], 2.0);
					}
				}
				pooling3[t][i][j] = sqrt(pooling3[t][i][j]);
			}
		}
	}
}

//conv4 and tanh5
void cnn::forward45()
{
	for(int t=0; t<32; ++t)
	{
		for(int i=0; i<10; ++i)
		{
			for(int j=0; j<10; ++j)
			{
				conv4tanh5[t][i][j] = 0;
				for(int f=0; f<16; ++f)
				{
					for(int m=0; m<5; ++m)
					{
						for(int n=0; n<5; ++n)
						{
							conv4tanh5[t][i][j] += pooling3[f][i+m][j+n] * conv4Weight[t][f][m][n];
						}
					}
				}
				conv4tanh5[t][i][j] += conv4Bias[t];
				conv4tanh5[t][i][j] = tanh(conv4tanh5[t][i][j]);
			}
		}
	}
}

//pooling6
void cnn::forward6()
{
	for(int t=0; t<32; ++t)
	{
		for(int i=0; i<5; ++i)
		{
			for(int j=0; j<5; ++j)
			{
				pooling6[t][i][j] = 0;
				for(int m=0; m<2; ++m)
				{
					for(int n=0; n<2; ++n)
					{
						pooling6[t][i][j] += pow(conv4tanh5[t][2*i+m][2*j+n], 2.0);
					}
				}
				pooling6[t][i][j] = sqrt(pooling6[t][i][j]);
			}
		}
	}
}

//reshape 7
void cnn::forward7()
{
	int idx = 0;
	for(int t=0; t<32; ++t)
	{
		for(int i=0; i<5; ++i)
		{
			for(int j=0; j<5; ++j)
			{
				reshape7[idx++] = pooling6[t][i][j];
			}
		}
	}
}

//linear8 and tanh9
void cnn::forward89()
{
	for(int t=0; t<256; ++t)// to layer
	{
		linear8tanh9[t] = 0;
		for(int f=0; f<800; ++f)// from layer
		{
			linear8tanh9[t] += reshape7[f] * linear8Weight[t][f]; 
		}
		linear8tanh9[t] += linear8Bias[t];
		linear8tanh9[t] = tanh(linear8tanh9[t]);
	}
}

//linear10
void cnn::forward10()
{
	for(int t=0; t<43; ++t)
	{
		linear10[t] = 0;
		for(int f=0; f<256; ++f)
		{
			linear10[t] += linear8tanh9[f] * linear10Weight[t][f];
		}
		linear10[t] += linear10Bias[t];
	}
}

int cnn::forward()
{
	forward12();
	forward3();
	forward45();
	forward6();
	forward7();
	forward89();
	forward10();

	double maxVal = linear10[0];
	int maxIdx = 0;
	for(int i=0; i<43; ++i)
	{
		if(linear10[i] > maxVal)
		{
			maxVal = linear10[i];
			maxIdx = i;
		}
	}

	classLabel = maxIdx + 1;//because the index of the label data is from 1
	
	return classLabel;
}
