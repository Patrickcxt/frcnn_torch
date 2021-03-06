require 'loadcaffe'
require 'image'
require 'ROIPooling'

function get_model()

---------------Convolutional Layers + ROI pooling--------------------------

  pretrained_model = loadcaffe.load( 'models/VGG_CNN_M_1024/test.prototxt', 'fast_rcnn_models/vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel' )
  -- print( pretrained_model )
  
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
  regression:add(nn.Linear(1024,84))
  rout = regression(node)

  cls_reg = nn.gModule( {fc_input}, {cout,rout} )

  return conv_ROI, cls_reg
end

