function weight = cnnWeight(weightFile)
% read the CNN weight from the file

fid = fopen(weightFile);
lineData = fscanf(fid, '%f',[1,inf]);
fclose(fid);
idx = 0;

% conv 1 weight 16*1*5*5
conv1Weight = zeros(5,5,16,1); % easy to get the sub matrix in this model
for i = 1:16
    for j = 1:1
        for m = 1:5
            for n = 1:5
                idx = idx +1;
                conv1Weight(m,n,i,j) = lineData(idx);
            end
        end
    end
end

% conv 1 bias 16
for i = 1:16
    idx = idx + 1;
    conv1Bias(i) = lineData(idx);
end

% conv 4 weight 32*16*5*5
conv4Weight = zeros(5,5,32,16); % easy to get the sub matrix in this model
for i = 1:32
    for j = 1:16
        for m = 1:5
            for n = 1:5
                idx = idx +1;
                conv4Weight(m,n,i,j) = lineData(idx);
            end
        end
    end
end

% conv 4 bias 32
for i = 1:32
    idx = idx + 1;
    conv4Bias(i) = lineData(idx);
end

% linear 8 weight 256*800
for i = 1:256
    for j = 1:800
        idx = idx + 1;
        linear8Weight(i,j) = lineData(idx);
    end
end

% linear 8 bias 256
for i = 1:256
    idx = idx + 1;
    linear8Bias(i) = lineData(idx);
end

% linear 10 weight 43*256
for i = 1:43
    for j = 1:256
        idx = idx + 1;
        linear10Weight(i,j) = lineData(idx);
    end
end

% linear 10 bias 43
for i = 1:43
    idx = idx + 1;
    linear10Bias(i) = lineData(idx);
end

% merge all weight
weight.conv1Weight = conv1Weight;
weight.conv1Bias = conv1Bias;
weight.conv4Weight = conv4Weight;
weight.conv4Bias = conv4Bias;
weight.linear8Weight = linear8Weight;
weight.linear8Bias = linear8Bias;
weight.linear10Weight = linear10Weight;
weight.linear10Bias = linear10Bias;
