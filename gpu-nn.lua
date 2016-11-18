require 'cutorch'
require 'cunn'
 
cutorch.setDevice(1)
--[[
model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 3, 3, 5))
model:add(nn.ReLU(true))
model:add(nn.SpatialConvolution(3, 3, 3, 5))
model:add(nn.ReLU(true))
model:cuda()
]]
local t = require 'Volumetric-model'
local model = t.model --expect input batch x nPlane x time x w x h

gpus = torch.range(1, cutorch.getDeviceCount()):totable()
dpt = nn.DataParallelTable(1):add(model, gpus):cuda()
 
input = torch.round(torch.CudaTensor(1, 3, 75, 50, 100):uniform(0, 255))
output = dpt:forward(input)
fakeGradients = output:clone():uniform(-0.1, 0.1)
dpt:backward(input, fakeGradients)
--x, dl_dx =dpt:getParameters()
print(dpt)
print(x)
