require 'ROIDetection'
require 'config'
require 'pascal_voc'
require 'torch'
require 'image'
require 'nms'
require 'utils'


-- example for ROIDetection
year = '2007'
data_type = 'train'
-- gt_bboxes, gt_classes = pascal_voc.get_gt_bboxes(year, data_type)
gt_bboxes, gt_classes = pascal_voc.get_gt_bboxes()

math.randomseed(tostring(os.time()):reverse():sub(1, 6))
len = gt_bboxes[2059]:size(1)
scores = torch.Tensor(len):zero()
scores:apply(function()
    return math.random()
end)
temp = torch.Tensor(len, 5)
temp[{{}, {1}}] = scores
temp[{{}, {2, 5}}] = gt_bboxes[2059]
print(temp)
pick = utils.nms(temp, 0.0)
print(pick)
-- ss_rois = pascal_voc.get_ss_rois(year, data_type)
--[=[
ss_rois = pascal_voc.get_ss_rois()
means, stds, roi_set, bbox_targets = ROIDetection.add_bbox_regression_targets(ss_rois, gt_bboxes, gt_classes)
print('means')
print(means)
print('stds')
print(stds)
]=]


