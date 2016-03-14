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
require 'data_layer'


conv_ROI, cls_reg = get_model()

--cls_reg = torch.load( 'trained_cls_reg.bin', 'binary' )

conv_ROI:training()
cls_reg:training()

print ( conv_ROI:__tostring() )
print ( cls_reg:__tostring() )

----------------------------Variables' definition--------------------------------

optimMethod = optim.sgd  -- use the Stochastic gradient descent optimization
optimState = {
      learningRate = 1e-3,
      weightDecay = 0.0005,
      momentum = 0.9,
      learningRateDecay = 1e-7
} -- parameters sdg needs

cls_reg_parameters, cls_reg_gradParameters = cls_reg:getParameters()
criterion_cls = nn.CrossEntropyCriterion()--:cuda
criterion_reg = nn.SmoothL1Criterion()--:cuda

---------------------------------------------------------------------------------


-------------------------------Loading Dataset-----------------------------------

gt_bboxes, gt_classes = pascal_voc.get_gt_bboxes()
ss_rois = pascal_voc.get_ss_rois()
means, stds, roi_set, bbox_targets, max_overlaps = ROIDetection.add_bbox_regression_targets(ss_rois, gt_bboxes, gt_classes)
images = pascal_voc.load_image_set_index()
data_layer.set_roidb(images, roi_set, bbox_targets, max_overlaps)

saved = 0

---------------------------------------------------------------------------------

for idx=1, 40000  do

  local time = os.time()
  local im_blob, rois_blob, bbox_targets_blob, labels_blob = data_layer.get_next_minibatch()
  input = { im_blob, rois_blob } --Preprocessed Input

  conv_features = conv_ROI:forward( input )

  -----------------Definition of Opfunc used for optim.sdg----------------------

  local function LossAndGradient(x)
    if x ~= cls_reg_parameters then
      cls_reg_parameters:copy(x)
    end

    cls_reg_gradParameters:zero()
    cls_loss = 0
    reg_loss = 0
    fsize = conv_features:size(1)

    for i = 1, fsize do
      output = cls_reg:forward( conv_features[i] )

      cls_err = criterion_cls:forward( output[1], labels_blob[i] )
      cls_loss = cls_loss + cls_err
      cls_delta = criterion_cls:backward( output[1], labels_blob[i] )

      reg_err = criterion_reg:forward( output[2], bbox_targets_blob[i] )
      reg_loss = reg_loss + reg_err
      reg_delta = criterion_reg:backward( output[2], bbox_targets_blob[i] )
      print( bbox_targets_blob[i] )

      cls_reg_delta = cls_reg:backward( conv_features[i], { cls_delta, reg_delta } )
    end

    cls_reg_gradParameters:div( fsize )
    pcls_loss = cls_loss/fsize
    preg_loss = reg_loss/fsize
    average_loss = pcls_loss + preg_loss
    return average_loss,cls_reg_gradParameters
  end
  ------------------------------------------------------------------------------

  optimMethod( LossAndGradient, cls_reg_parameters, optimState )

  taken_time = os.time() - time
  print( idx .. 'th batch  ' .. taken_time .. 's')
  ------------------------------------------------------------------------------

  local save_num = 1000
  if idx%save_num == 0 then
    torch.save( 'trained_cls_reg.t7', cls_reg, 'binary' )
    saved = saved + 1
    print( 'Saved ' .. saved .. ' times' )
  end
end
