require 'torch'
require 'nn'
require 'image'
require 'config'
require 'hdf5'

local utilities = paths.dofile('utils.lua')
local boxoverlap = utilities.boxoverlap
local file_exists = utilities.file_exists

ROIDetection = {}
local _EPS = config._EPS


local function _read_from_h5(cache_file)
    local myFile = hdf5.open(cache_file, 'r')
    local means = myFile:read('means'):all()
    local stds = myFile:read('stds'):all()
    local num_bboxes = myFile:read('num_bboxes'):all()[1]
    local bbox_targets = {}
    for i = 1, num_bboxes do 
        table.insert(bbox_targets, myFile:read('reg_bboxes_' .. tostring(i)):all())
    end
    myFile:close() 
    return means, stds, bbox_targets
end

local function _write_to_h5(cache_file, means, stds, bbox_targets)
    local myFile = hdf5.open(cache_file, 'w')
    myFile:write('means', means)
    myFile:write('stds', stds)
    num_tensor = torch.IntTensor({table.getn(bbox_targets)})
    myFile:write('num_bboxes', num_tensor)
    for i = 1, table.getn(bbox_targets) do
        myFile:write('reg_bboxes_' .. tostring(i), bbox_targets[i])
    end
    myFile:close()
end

local function _merge_rois(ss_boxes, gt_boxes)
    local num_ss_boxes = ss_boxes:dim() > 0 and ss_boxes:size(1) or 0
    local num_gt_boxes = gt_boxes:dim() > 0 and gt_boxes:size(1) or 0
    local total = num_gt_boxes + num_ss_boxes
    local rois = torch.DoubleTensor(total, 4)
    rois[{{1, num_gt_boxes}, {}}] = gt_boxes
    rois[{{num_gt_boxes+1, total}, {}}] = ss_boxes
    return rois
end

local function _compute_targets(ss_boxes, gt_boxes, gt_classes)
  ss_boxes:add(-1)
  gt_boxes:add(-1)
  local num_gt_boxes = gt_boxes:dim() > 0 and gt_boxes:size(1) or 0
  local num_ss_boxes = ss_boxes:dim() > 0 and ss_boxes:size(1) or 0
  local total = num_gt_boxes + num_ss_boxes
  local rois = _merge_rois(ss_boxes, gt_boxes)
  local overlap_class = torch.FloatTensor(total, 21):zero()
  local overlap = torch.FloatTensor(total,num_gt_boxes):zero()
  for idx=1,num_gt_boxes do
    local o = boxoverlap(rois,gt_boxes[idx])
    local tmp = overlap_class[{{},gt_classes[idx]}] -- pointer copy
    tmp[tmp:lt(o)] = o[tmp:lt(o)]
    overlap[{{},idx}] = o
  end
   
  -- ignore ROI whose overlas < cfg.TRAIN.BBOX_THRESH
  gt_val, gt_assignment = overlap:max(2)
  lb_val, label = overlap_class:max(2)
  gt_assignment = torch.squeeze(gt_assignment, 2)
  label = torch.squeeze(label, 2)
  gt_inds = torch.squeeze(lb_val:eq(1), 2)
  ex_inds = torch.squeeze(lb_val:ge(config.TRAIN_BBOX_THRESH), 2)

  -- Construct gt rois and examples rois for which we try to make predictions
  gt_rois = torch.DoubleTensor(ex_inds:sum(), 4)
  ex_rois = torch.DoubleTensor(ex_inds:sum(), 4)
  local j = 1
  for i = 1, total do
      if ex_inds[i] ~= 0 then
          ex_rois[j] = rois[i]
          gt_rois[j] = rois[gt_assignment[i]]
          j = j + 1
      end
  end

  ex_widths = (ex_rois[{{}, {3}}] - ex_rois[{{}, {1}}]):add(_EPS) -- + EPS
  ex_heights = (ex_rois[{{}, {4}}] - ex_rois[{{}, {2}}]):add(_EPS) -- + EPS
  ex_ctr_x = ex_rois[{{}, {1}}]:clone():add(ex_widths:clone():mul(0.5))
  ex_ctr_y = ex_rois[{{}, {2}}]:clone():add(ex_heights:clone():mul(0.5))

  gt_widths = (gt_rois[{{}, {3}}] - gt_rois[{{}, {1}}]):add(_EPS) -- + EPS
  gt_heights = (gt_rois[{{}, {4}}] -gt_rois[{{}, {2}}]):add(_EPS) -- + EPS
  gt_ctr_x = gt_rois[{{}, {1}}]:clone():add(gt_widths:clone():mul(0.5))
  gt_ctr_y = gt_rois[{{}, {2}}]:clone():add(gt_heights:clone():mul(0.5))
  targets_dx = (gt_ctr_x:add(-ex_ctr_x)):cdiv(ex_widths)
  targets_dy = (gt_ctr_y:add(-ex_ctr_y)):cdiv(ex_heights)
  targets_dw = torch.log(gt_widths:cdiv(ex_widths))
  targets_dh = torch.log(gt_heights:cdiv(ex_heights))
  
  targets = torch.zeros(total, 5)
  local j = 1
  for i = 1, total do
      if ex_inds[i] ~= 0 then
          targets[{{i}, {1}}] = label[i]
          targets[{{i}, {2}}] = targets_dx[j]
          targets[{{i}, {3}}] = targets_dy[j]
          targets[{{i}, {4}}] = targets_dw[j]
          targets[{{i}, {5}}] = targets_dh[j]
          j = j + 1
      else
          targets[{{i}, {1}}] = 21
      end
  end

  return rois, targets
end


function ROIDetection.add_bbox_regression_targets(ss_roi_set, gt_roi_set, gt_classes)
    -- Add information needed to train bounding-box regressors.
    
    print('Computing bounding-box regression targets...')
    
    local bbox_targets = {}
    local roi_set = {}
    local num_images = #ss_roi_set
    local num_classes = config.num_classes

    local cache_file = './data/cache/reg_voc_' .. config.year .. '_' .. config.data_type .. '.h5'
    if file_exists(cache_file) then
        -- read targets from existed cache file
        print('Loading from ' .. cache_file)
        means, stds, bbox_targets = _read_from_h5(cache_file)
        for im_i = 1, num_images do
            local ss_rois = ss_roi_set[im_i]:add(-1)
            local gt_rois = gt_roi_set[im_i]:add(-1)
            local rois = _merge_rois(ss_rois, gt_rois)
            table.insert(roi_set, rois)
        end
        for im_i = 1, num_images do
            print('image ' .. tostring(im_i) .. ': ')
            for j = 1, bbox_targets[im_i]:size(1) do
                if bbox_targets[im_i][j][2] == nan or bbox_targets[im_i][j][3]  == nan or bbox_targets[im_i][j][4] == nan or bbox_targets[im_i][j][5] == nan then
                    print(bbox_targets[im_i][j])
                    print('===================================yes============================================')
                end
            end
        end
        print('Done')
        return means, stds, roi_set, bbox_targets
    end

    for im_i = 1, num_images do
        local ss_rois = ss_roi_set[im_i]
        local gt_rois = gt_roi_set[im_i]
        local max_classes = gt_classes[im_i]
        local rois, targets = _compute_targets(ss_rois, gt_rois, max_classes)
        table.insert(bbox_targets, targets)
        table.insert(roi_set, rois)
    end

    -- Compute values needed for means and std
    -- var(x) = E(x^2) - E(x)^2
    local class_counts = torch.zeros(num_classes, 1):add(_EPS)
    local sums = torch.zeros(num_classes, 4)
    local squared_sums = torch.zeros(num_classes, 4)
    for im_i = 1, num_images do
        -- print('image ' .. tostring(im_i) .. ': ')
        local targets = bbox_targets[im_i]:clone()
        for cls = 1, num_classes-1 do
            local tmp = torch.squeeze(targets[{{}, {1}}])
            local inds = tmp:eq(cls)
            local cls_set = torch.zeros(inds:sum(), 4)
            local num = 0
            for j = 1, targets:size(1) do
                if inds[j] ~= 0 then 
                    num = num + 1
                    cls_set[num]:copy(targets[{{j}, {2, 5}}])
                end
            end
            if  num > 0 then
                class_counts[cls]:add(num)
                sums[{{cls}, {}}]:add(cls_set:sum(1))
                squared_sums[{{cls}, {}}]:add((cls_set:pow(2)):sum(1))
            end
        end
    end

    -- means = sums / class_counts
    local means = sums
    local stds = squared_sums
    for i = 1, 4 do
        means[{{}, {i}}]:cdiv(class_counts)
        stds[{{}, {i}}]:cdiv(class_counts)
    end
    local tmp = torch.zeros(num_classes, 4):copy(means)
    -- stds = np.sqrt(squared_sums / class_counts - means ** 2)
    stds:add(-(tmp:pow(2))):sqrt()

    -- Normalize targets
    for im_i = 1, num_images do
         print('image ' .. tostring(im_i) .. ': ')
        local targets = bbox_targets[im_i]
        for j = 1, targets:size(1) do
            local cls = torch.round(targets[j][1])
            if cls ~= 21 then
                bbox_targets[im_i][{{j}, {2, 5}}]:add(-means[{{cls}, {}}])
                bbox_targets[im_i][{{j}, {2, 5}}]:cdiv(stds[{{cls}, {}}])
            end
        end
        --print(bbox_targets[im_i])
        --io.read()
    end
    print('Writing to ' .. cache_file)
    _write_to_h5(cache_file, means, stds, bbox_targets)
    print('Done')
    -- These values will be needed for making predictions
    -- (The predicts will need to be unnormalized and uncentered)
    return means, stds, roi_set, bbox_targets
end


return ROIDetection

