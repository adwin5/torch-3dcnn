--require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'rnn'

-- input size is batchSize x nPlane x time x w x h, 50 x 3 x 75 x 50 x 100
function deep_model_4d(time)
 local cnn = nn.Sequential();
 print('creating Volumetric model')

 -- first convolution, non-linear, and pooling
 cnn:add(nn.VolumetricConvolution(3,32,3,5,5,1,2,2,1,2,2)) 
 cnn:add(nn.ReLU()) 
 cnn:add(nn.VolumetricMaxPooling(1,2,2,1,2,2)) 
 cnn:add(nn.VolumetricDropout(0.2))
 --print("cnn done")

 --first convolution, non-linear, and pooling
 cnn:add(nn.VolumetricConvolution(32,64,3,5,5,1,1,1,1,2,2)) 
 cnn:add(nn.ReLU()) 
 cnn:add(nn.VolumetricMaxPooling(1,2,2,1,2,2)) 
 cnn:add(nn.VolumetricDropout(0.2))
 --print("cnn second done")

 --first convolution, non-linear, and pooling
 cnn:add(nn.VolumetricConvolution(64,96,3,5,5,1,1,1,1,2,2)) 
 cnn:add(nn.ReLU()) 
 cnn:add(nn.VolumetricMaxPooling(1,2,2,1,2,2)) 
 cnn:add(nn.VolumetricDropout(0.2))
 --print("cnn third done")
 
 --Expects an input shape of batchsize x seqLen x inputsize By setting [batchFirst] to true
 cnn:add(nn.Transpose({2,3}))-- from batch x channel x len x width x height > batch x len x channel x width x height
 cnn:add(nn.View(1,time,96*3*6):setNumInputDims(5))--from batch x len x channel x width x height > batch x len x (channel x width x height)
 brnn = nn.SeqBRNN(96*3*6, 256, true)
 cnn:add(brnn)

 --TODO upsampling

 --Linear
 cnn:add(nn.Reshape(1*time*256))
 cnn:add(nn.Linear(time*256, time*128))
 cnn:add(nn.ReLU())
 cnn:add(nn.Linear(time*128, 10)) --mock for mnist case
 cnn:add(nn.LogSoftMax())

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
print("model loaded")
print(model)
--print(#model:parameters()[1])
x, dl_dx = model:getParameters()
print("number of parameer: " .. x:size(1))

step=function()
    local input = torch.CudaTensor(1,3,time,50,100)
    local inputs=input:clone()

    local shuffle = torch.randperm(trainset.size)    
    local target = trainset.label[shuffle[1]]
    local targets = torch.CudaTensor(1)
    targets[1] = target
    targets:add(1)
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
    return fs[1]
end
loss=step()
i=1
print(string.format('Epoch: %d Current loss: %4f', i, loss))

-- return package:
return {
  model = model
   --loss = loss,
}
