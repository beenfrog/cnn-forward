--convert the cnn weight and bias in the  'gtsrb.net' to txt file

require 'torch'
require 'nn'
require 'image'

--load the torch result file
net = torch.load('gtsrb.net')

--image.display{image=net.modules[1].weight:resize(16,5,5),padding=2,zoom=10,nrow=4,legend='weight1'}
--image.display{image=net.modules[4].weight:resize(512,5,5),padding=2,zoom=10,nrow=16,legend='weight4'}

-----------------------------------------------------
--save the net to txt
file=io.open('gtsrb.txt', 'w')
assert(file)

-----------------------------------------------------
--conv(1) weight
for i=1,16 do
	for j=1,25 do
		file:write(net.modules[1].weight[i][j],' ')
	end
end
file:write('\n')

--conv(1) bais
for i=1,16 do
	file:write(net.modules[1].bias[i],' ')
end
file:write('\n')

-----------------------------------------------------
--conv(4) weight
for i=1,32 do
	for j=1,400 do
		file:write(net.modules[4].weight[i][j],' ')
	end
end
file:write('\n')

--conv(4) bais
for i=1,32 do
	file:write(net.modules[4].bias[i],' ')
end
file:write('\n')

-----------------------------------------------------
--Linear(8) weight
for i=1,256 do
	for j=1,800 do
		file:write(net.modules[8].weight[i][j],' ')
	end
end
file:write('\n')

--Linear(8) bias
for i=1,256 do
	file:write(net.modules[8].bias[i],' ')
end
file:write('\n')

-----------------------------------------------------
--Linear(10) weight
for i=1,43 do
	for j=1,256 do
		file:write(net.modules[10].weight[i][j],' ')
	end
end
file:write('\n')

--Linear(10) bias
for i=1,43 do
	file:write(net.modules[10].bias[i],' ')
end
file:write('\n')


-----------------------------------------------------
file:close()
