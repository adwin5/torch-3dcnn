--require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'

-- input size is batchSize x nPlane x time x w x h, 50 x 3 x 75 x 50 x 100
function deep_model_4d(time)
 local cnn = nn.Sequential();
 print('creating Volumetric model')

 -- first convolution, non-linear, and pooling
 cnn:add(nn.VolumetricConvolution(3,32,3,5,5,1,2,2,1,2,2)) 
 cnn:add(nn.ReLU()) 
 cnn:add(nn.VolumetricMaxPooling(1,2,2,1,2,2)) 
 cnn:add(nn.VolumetricDropout(0.2))

 -- second convolution, non-linear, and pooling
 cnn:add(nn.VolumetricConvolution(32,64,3,5,5,1,1,1,1,2,2)) 
 cnn:add(nn.ReLU()) 
 cnn:add(nn.VolumetricMaxPooling(1,2,2,1,2,2)) 
 cnn:add(nn.VolumetricDropout(0.2))

 -- third convolution, non-linear, and pooling
 cnn:add(nn.VolumetricConvolution(64,96,3,3,3,1,1,1,1,1,1))
 cnn:add(nn.ReLU())
 cnn:add(nn.VolumetricMaxPooling(1,2,2,1,2,2))
 cnn:add(nn.VolumetricDropout(0.2))

 -- Bi LSTM
 -- hfiddenSize=128
 -- lstmp = nn.SeqLSTMP(96*3*6, hiddensize, 256)
 -- cnn:add(lstmp)

 -- linear
 cnn:add(nn.Reshape(time*96*3*6))
 cnn:add(nn.Linear(time*96*3*6, time*128))
 cnn:add(nn.ReLU()) 
 cnn:add(nn.Linear(time*128, time*28))
 cnn:add(nn.ReLU())--code for mnist case 
 cnn:add(nn.Linear(time*28, 10))--code for mnist case
 cnn:add(nn.LogSoftMax())

 --convert to cudnn
 --cudnn.convert(cnn, cudnn)
 return cnn:cuda()
end

--test file
---[[
--nPlane x time x w x h
mnist = require 'mnist'
require 'optim'
fullset = mnist.traindataset()
trainset = {
    size = 3,
    label = fullset.label[{{1,500}}]
}

criterion = nn.ClassNLLCriterion():cuda()
sgd_params = {
   learningRate = 1e-2,
   learningRateDecay = 1e-4,
   weightDecay = 1e-3,
   momentum = 1e-4
}
--]]
time=75
model=deep_model_4d(time)
---[[
x, dl_dx = model:getParameters()
local input = torch.CudaTensor(1,3,time,50,100)
--local y=model:forward(input)
--print(y:size())
local inputs=input:clone()

target = trainset.label[1]
targets = torch.CudaTensor(1)
targets[1] = target
targets:add(1)
print(targets)
--print(#model:parameters()[1])
print(x:size(1))
local feval = function(x_new)
    -- reset data
    if x ~= x_new then x:copy(x_new) end
    dl_dx:zero()

    -- perform mini-batch gradient descent
    local loss = criterion:forward(model:forward(inputs), targets)
    local tmp_diff = criterion:backward(model.output, targets)
    model:backward(inputs, tmp_diff)
    print(model.output:size())
    return loss, dl_dx
end
_, fs = optim.sgd(feval, x, sgd_params)
print(fs)
--x, dl_dx = model:getParameters()
--print(model)
--]]

-- return package:
return {
  model = model
   --loss = loss,
}
