---------------------------------
-- Author: Weimian Li
-- Date:   2016.1-present
-----------------------------------

require 'torch'
require 'nn'
require 'nngraph'
require 'ROIPooling'
require 'loadcaffe'
require 'optim'
require 'image'
require 'ROIDetection'
require 'pascal_voc'
require 'model'


conv_ROI, cls_reg = get_model()
conv_ROI:training()
cls_reg:training()
print ( conv_ROI:__tostring() )
print ( cls_reg:__tostring() )
io.read()
nngraph.display(cls_reg)
io.read()

----------------------------Variables' definition--------------------------------

optimMethod = optim.sgd  -- use the Stochastic gradient descent optimization
optimState = {
      learningRate = 1e-3,
      weightDecay = 0.0005,
      momentum = 0.9,
      learningRateDecay = 1e-7
} -- parameters sdg needs

cls_reg_parameters, cls_reg_gradParameters = cls_reg:getParameters()
criterion_cls = nn.ClassNLLCriterion()
criterion_reg = nn.SmoothL1Criterion()

roi_batch = 128
img_batch = 1
---------------------------------------------------------------------------------


-------------------------------Loading Dataset-----------------------------------

Year = '2007'
Type = 'train'

train_images = pascal_voc.load_image_set_index( Year, Type )
all_gt_bboxes, all_gt_classes = pascal_voc.get_gt_bboxes( Year, Type )
all_ss_rois = pascal_voc.get_ss_rois( Year,Type )
means, stds, rois, bbox_targets = ROIDetection.add_bbox_regression_targets( all_ss_rois, all_gt_bboxes, all_gt_classes ) 

img_num = #train_images
randIdx = torch.randperm( img_num )

---------------------------------------------------------------------------------


for idx=1, img_num, img_batch do

  local time = os.time()
  ------------------------------------------------------------------------------
  Image = torch.Tensor( math.min(img_batch,img_num-idx+1), 3, 224, 224 )

  roi_num_perbatch = 0
  rois_perbatch = {}
  targets_perbatch = {}

  for i=idx, math.min( idx+img_batch-1, img_num ) do
    img = pascal_voc.get_image( train_images[ randIdx[i] ], Year )
    ss_rois = rois[ randIdx[i] ]

    ------------------Resize image and rois------------------------

    img = image.scale( img, 224, 224 )
    local img_size = img:size()
    local swidth = 224/img_size[3]
    local sheight = 224/img_size[2]
    ss_rois[{ {}, {1} }]:mul(swidth):round()
    ss_rois[{ {}, {3} }]:mul(swidth):round()
    ss_rois[{ {}, {2} }]:mul(sheight):round()
    ss_rois[{ {}, {4} }]:mul(sheight):round()

    rois_imgIdx = torch.Tensor( ss_rois:size(1), 5 )
    rois_imgIdx[{ {},{1} }] = (i-1)%img_batch + 1
    rois_imgIdx[{ {},{2,5} }] = ss_rois

    ---------------------------------------------------------------
    table.insert( rois_perbatch, rois_imgIdx )
    table.insert( targets_perbatch, bbox_targets[ randIdx[i] ] )
    roi_num_perbatch = roi_num_perbatch + ss_rois:size(1)
    Image[ (i-1)%img_batch+1 ] = img
  end

  ROIs = torch.Tensor( roi_num_perbatch, 5 )
  Targets = torch.Tensor( roi_num_perbatch, 5 )
  local index = 1
  for j = 1, #rois_perbatch do
    ROIs[{ {index,index+rois_perbatch[j]:size(1)-1}, {} }] = rois_perbatch[j]
    Targets[{ {index,index+rois_perbatch[j]:size(1)-1}, {} }] = targets_perbatch[j]
    index = index + rois_perbatch[j]:size(1)
  end

  input = { Image, ROIs } --Preprocessed Input
  -----------------------------------------------------------------------------

  conv_features = conv_ROI:forward( input )
  train_size = conv_features:size(1)
  shuffle = torch.randperm( train_size )

  for t = 1, train_size, roi_batch do
    --create mini batch
    fc_inputs = {}
    cls_targets = {}
    roi_targets = {}
    for i = t, math.min( t + roi_batch - 1, train_size ) do
      each_sample = conv_features[ shuffle[i] ]
      each_cls_target = Targets[ shuffle[i] ][1]
      each_roi_target = Targets[ shuffle[i] ][{{2,5}}]
      table.insert( fc_inputs, each_sample )
      table.insert( cls_targets, each_cls_target )
      table.insert( roi_targets, each_roi_target )
    end

    -----------------Definition of Opfunc used for optim.sdg----------------------

    local function LossAndGradient(x)
      if x ~= cls_reg_parameters then
        cls_reg_parameters:copy(x)
      end

      cls_reg_gradParameters:zero()
      cls_loss = 0
      reg_loss = 0

      for i = 1, #fc_inputs do
        output = cls_reg:forward( fc_inputs[i] )

        cls_err = criterion_cls:forward( output[1], cls_targets[i] )
        cls_loss = cls_loss + cls_err
        cls_delta = criterion_cls:backward( output[1], cls_targets[i] )

        reg_err = criterion_reg:forward( output[2], roi_targets[i] )
        reg_loss = reg_loss + reg_err
        reg_delta = criterion_reg:backward( output[2], roi_targets[i] )

        cls_reg_delta = cls_reg:backward( fc_inputs[i], { cls_delta, reg_delta } )
      end

      cls_reg_gradParameters:div( #fc_inputs )
      pcls_loss = cls_loss/#fc_inputs
      preg_loss = reg_loss/#fc_inputs
      average_loss = pcls_loss + preg_loss
      return average_loss,cls_reg_gradParameters
    end
    ------------------------------------------------------------------------------

    local x,fx = optimMethod( LossAndGradient, cls_reg_parameters, optimState )
  end

  taken_time = os.time() - time
  print( idx+img_batch-1 .. '/' .. img_num .. '    rois: ' .. train_size .. '     ' .. taken_time .. 's' )
  ------------------------------------------------------------------------------
end

conv_ROI:evaluate()
cls_reg:evaluate()

torch.saveobj( 'trained_conv_ROI.bin', conv_ROI, 'binary' )
torch.saveobj( 'trained_cls_reg.bin', cls_reg, 'binary' )

