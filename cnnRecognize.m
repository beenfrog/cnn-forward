function label = cnnRecognize(img, weight)
% Recognize the image 

% using the image of double
img = im2double(img);

% conv1 and tanh2
conv1tanh2 = zeros(28,28,16);
for t = 1:16
    for f = 1:1
        conv1tanh2(:,:,t) = conv1tanh2(:,:,t) + filter2(weight.conv1Weight(:,:,t,f), img, 'valid');
    end
    conv1tanh2(:,:,t) = conv1tanh2(:,:,t) + weight.conv1Bias(t) * ones(28,28);
    conv1tanh2(:,:,t) = tanh(conv1tanh2(:,:,t));
end

% pooling3
pooling3 = zeros(14,14,16);
for t = 1:16
    for i = 1:14
        for j = 1:14
            vector = reshape(conv1tanh2(2*i-1 : 2*i, 2*j-1 : 2*j, t) ,[1,4]);
            pooling3(i, j, t) = norm(vector, 2);
        end
    end
end

% conv4 and tanh5
conv4tanh5 = zeros(10,10,32);
for t = 1:32
    for f = 1:16
        conv4tanh5(:,:,t) = conv4tanh5(:,:,t) + filter2(weight.conv4Weight(:,:,t,f), pooling3(:,:,f), 'valid');
    end
    conv4tanh5(:,:,t) = conv4tanh5(:,:,t) + weight.conv4Bias(t) * ones(10,10);
    conv4tanh5(:,:,t) = tanh(conv4tanh5(:,:,t));
end

% pooling6
pooling6 = zeros(5,5,32);
for t = 1:32
    for i = 1:5
        for j = 1:5
            vector = reshape(conv4tanh5(2*i-1 : 2*i, 2*j-1 : 2*j, t) ,[1,4]);
            pooling6(i, j, t) = norm(vector, 2);
        end
    end
end

% reshape 7
idx = 0;
reshape7 = zeros(800,1);
for t = 1:32
    for i = 1:5
        for j = 1:5
            idx = idx + 1;
            reshape7(idx) = pooling6(i,j,t);
        end
    end
end

% linear8 and tanh9
linear8tanh9 = zeros(256,1);
for t = 1:256
    for f = 1:800
        linear8tanh9(t) = linear8tanh9(t) + reshape7(f) * weight.linear8Weight(t,f);
    end
    linear8tanh9(t) = linear8tanh9(t) + weight.linear8Bias(t);
    linear8tanh9(t) = tanh( linear8tanh9(t) );
end

% linear10
linear10 = zeros(43,1);
for t = 1:43
    for f = 1:256
        linear10(t) = linear10(t) + linear8tanh9(f) * weight.linear10Weight(t,f);
    end
    linear10(t) = linear10(t) + weight.linear10Bias(t);
end
        
% recognize accrossing the max element in the last layer
[~, label] = max(linear10);