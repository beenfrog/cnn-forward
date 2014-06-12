% run the test imgae

clear;clc
weight = cnnWeight('./data/gtsrb.txt');

tic
for i = 1:99
    img = imread(['./data/png/', num2str(i,'%05d.png')]);
    label(i) = cnnRecognize(img, weight);
end
toc 

label'