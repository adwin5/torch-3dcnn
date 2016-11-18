--------------------------------------------------------------------------------
-- loading of a video for torch7
--------------------------------------------------------------------------------
-- Jonghoon, Jin, E. Culurciello, October 9, 2014
--------------------------------------------------------------------------------

require("pl")
require("image")
local video = assert(require("libvideo_decoder"))

-- Options ---------------------------------------------------------------------
opt = lapp([[
-v, --videoPath    (default '')    path to video file
]])


-- load a video and extract frame dimensions
local status, height, width, length, fps = video.init(opt.videoPath)
if not status then
   error("No video")
else
   print('Video statistics: '..height..'x'..width..' ('..(length or 'unknown')..' frames)')
end

-- construct tensor
local dst = torch.ByteTensor(3, height, width)
function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

-- looping the video for 'length' times
local win
local nb_frames = length or fps*3
local dst_4d=torch.ByteTensor(1,3,height,width)
local dst_out
local timer = torch.Timer()
for i = 0, nb_frames-1 do
   video.frame_rgb(dst)
   dst_tmp=dst:clone()
   dst_4d=dst_tmp:resize(1,3,height,width)

   if i==0 then--create 4d output tensor
     dst_out=dst_4d:clone() 
   else
     dst_out=torch.cat(dst_out,dst_4d,1)
   end
   --print(dst_out:size())
   --display
   --win = image.display{image = dst, win = win}
   --sleep(0.05)
end
print(dst_out:size())
print(nb_frames)
print(fps)
print('Time: ', timer:time().real/nb_frames)

-- free variables and close the video
video.exit()

---[[--change shape
dst_out=dst_out:permute(2,1,3,4)
print(dst_out:size())
dst_out=dst_out:resize(3,75,50,100)
print(dst_out:size())
--]]

--import model from Volumetric-model.th
local t = require 'Volumetric-model'
local model = t.model --expect input --nPlane x time x w x h
print(model)

--[[--print out test forwarding
y=model:forward(dst_out:cuda())
print(y:size())
print(y)
--]]

--mock dataset
mnist = require 'mnist'
require 'optim'
require 'cutorch'
require 'cudnn'
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
x, dl_dx = model:getParameters()

-- put one sample each time first
step = function(batch_size)
    local current_loss = 0
    local count = 0
    local shuffle = torch.randperm(trainset.size)
    batch_size = batch_size or 200
    
    for t = 1, trainset.size, batch_size do
        -- setup inputs and targets for this mini-batch
        local size = math.min(t + batch_size - 1, trainset.size) - t
        local inputs = torch.CudaTensor(size, 3,75,50,100)
        local targets = torch.CudaTensor(size)
        for i = 1,size do
            --local input = trainset.data[shuffle[i+t]]
            local input = torch.CudaTensor(3,75,50,100)
            local target = trainset.label[shuffle[i+t]]
            -- if target == 0 then target = 10 end
            inputs[i] = input
            targets[i] = target:cuda()
        end
        targets:add(1)
        print(inputs:size())
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
        -- fs is a table containing value of the loss function
        -- (just 1 value for the SGD optimization)
        count = count + 1
        current_loss = current_loss + fs[1]
    end

    -- normalize loss
    return current_loss / count
end
--print(step(2))

max_iters = 2
do
    local last_accuracy = 0
    local decreasing = 0
    local threshold = 1 -- how many deacreasing epochs we allow
    for i = 1,max_iters do
        local loss = step()
        print(string.format('Epoch: %d Current loss: %4f', i, loss))
        local accuracy = eval(validationset)
        print(string.format('Accuracy on the validation set: %4f', accuracy))
        if accuracy < last_accuracy then
            if decreasing > threshold then break end
            decreasing = decreasing + 1
        else
            decreasing = 0
        end
        last_accuracy = accuracy
    end
end
