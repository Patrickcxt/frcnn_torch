require 'loadcaffe'
require 'image'


function get_model()

---------------Convolutional Layers + ROI pooling--------------------------
  --pretrained_model = loadcaffe.load( 'VGG_CNN_S_deploy.prototxt', 'VGG_CNN_S.caffemodel' )

  pretrained_model = loadcaffe.load( 'VGG_CNN_M_1024_deploy.prototxt', 'VGG_CNN_M_1024.caffemodel' )
  --print( pretrained_model )
  for i=24,15,-1 do
    pretrained_model:remove(i)
  end
  conv = pretrained_model --use the pretrained convs to decrease training time

  prl = nn.ParallelTable()
  prl:add(conv)
  prl:add(nn.Identity())
  ROIPooling = nn.ROIPooling(6, 6):setSpatialScale(1/16)

  conv_ROI = nn.Sequential()
  conv_ROI:add(prl)
  conv_ROI:add(ROIPooling)
  conv_ROI:add(nn.View(-1):setNumInputDims(3))

  -------------------------Fully Connected Network-----------------------------

  fc = nn.Sequential()

  -- fc6
  fc:add(nn.Linear(18432, 4096))
  fc:add(nn.ReLU(true))
  fc:add(nn.Dropout(0.500000))

  -- fc7
  fc:add(nn.Linear(4096, 1024))
  fc:add(nn.ReLU(true))
  fc:add(nn.Dropout(0.500000))

  ---------From here the network split into classifier and bbx regression---------

  fc_input = nn.Identity()()
  node = fc(fc_input)


  classifier = nn.Sequential()
  classifier:add(nn.Linear(1024,21))
  cout = classifier(node)

  regression = nn.Sequential()
  regression:add(nn.Linear(1024,4))
  rout = regression(node)

  cls_reg = nn.gModule( {fc_input}, {cout,rout} )

  --[[
  local function init(module, name)
    local function init_module(m)
      for k,v in pairs(m:findModules(name)) do
        local n = v.kW * v.kH * v.nOutputPlane
        v.weight:normal(0, math.sqrt(2 / n))
        v.bias:zero()
      end
    end
    module:apply(init_module)
  end

  init(cls_reg, 'nn.SpatialConvolution')
]]
  return conv_ROI, cls_reg
end
