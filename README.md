cnn-forward
===========

Using the CNN trained by Torch7 to classify images in: 1). C++ with the support of OpenCV; 2). Matlab.

#Dependency
1. OpenCV is required to read the images.
2. Torch7 is required to convert the cnn result(eg. gstrb.net) to txt file.
3. Matlab is required run the Maltab version of the code.

#How to run the program
The porgram is writted and test  in Debian GNU/Linux, it is easy to run in other system
```
#for c++
cd forward-cnn
cmake .
make
./run

#for matlab
cd forward-cnn
run('cnnRun.m')
```
#The structure of the cnn in torch
```
model:add(nn.SpatialConvolutionMM(1, 16, 5, 5))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(16,2, 2, 2, 2, 2))

model:add(nn.SpatialConvolutionMM(16, 32, 5, 5))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(32,2,2, 2, 2, 2))

model:add(nn.Reshape(32*5*5))
model:add(nn.Linear(32*5*5, 256))
model:add(nn.Tanh())
model:add(nn.Linear(256, 43))
```

#About the files
- `./data/torch-net-to-txt.lua` is used to convert the weight and bias in gtsrb.net to txt file.
- `./data/png`:some png images are here for testing
- `./data/gtsrb.net` is trained by Torch7
- `cnn.cpp cnn.h`: forward cnn class written in c++
- `run.cpp`: test program
- `*.m`: matlab file