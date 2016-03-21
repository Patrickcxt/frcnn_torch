
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
require 'config'
require 'data_layer'
require 'hdf5'


function be_print(tb, name)
    -- For debug ...
    print('=========================== ' .. name .. ' ===================================')
    -- io.read()
    local s = ''
    for i = 1, tb:size(1) do
        s  = s .. tb[i] .. ' '
    end
    print(s)
end

function my_print(tb, name)
    -- For debug ...
    print('=========================== ' .. name .. ' ===================================')
    io.read()
    for i = 1, 5 do
        print(tb[i])
    end
end

local cls_names = {
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor',
    '__background__'
}
local images = pascal_voc.load_image_set_index()

local function _get_image_blob(im)
    -- Converts an image into a network input
    -- Return:
    --     blob: a data blob holding an image pyramid
    --     im_scale_factors(list): list of image scales (relative to im) used
    im, im_scales =  data_layer.prep_im_for_blob(im, config.PIXEL_MEANS, config.TEST_SCALES[1], config.TEST_MAX_SIZE)
    local im_blob = data_layer.im_list_to_blob({im})
    return im_blob, {im_scales}
end

local function _get_rois_blob(im_rois, im_scales_factors)
    -- Convert RoIs into network inputs.
    -- Returns:
    --     blob: R x 5 matrix of RoIs in the image pyramid
    local rois  = data_layer.project_im_rois(im_rois, im_scales_factors[1])
    local rois_blob = torch.zeros(rois:size(1), 5)
    rois_blob[{{}, {1}}]:fill(1)
    rois_blob[{{}, {2, 5}}] = rois
    return rois_blob
end


local function _get_blobs(im, im_rois)
    -- Convert an image and RoIs within that iamge into network inputs.
    local data, im_scale_factors = _get_image_blob(im)
    local rois = _get_rois_blob(im_rois, im_scale_factors)
    local input_blob = {data, rois}
    return input_blob, im_scale_factors
end

local function _bbox_pred(boxes, box_deltas)
    -- Transform the set of class-agnostic boxes into class-specific boxes by applying the predicted offsets (box_deltas))
    box_deltas:double()
    if boxes:size(1) == 0 then
        return torch.zeros(0, box_deltas:size(2))
    end
    local widths = boxes[{{}, {3}}] - boxes[{{}, {1}}] + config._EPS
    local heights = boxes[{{}, {4}}] - boxes[{{}, {2}}] + config._EPS
    local ctr_x = boxes[{{}, {1}}] + widths * 0.5
    local ctr_y = boxes[{{}, {2}}] + heights * 0.5

    local pred_boxes = torch.zeros(box_deltas:size())
    for i = 1, 84, 4 do
        local dx = box_deltas[{{}, {i}}]
        local dy = box_deltas[{{}, {i+1}}]
        local dw = box_deltas[{{}, {i+2}}]
        local dh = box_deltas[{{}, {i+3}}]

        local pred_ctr_x = torch.cmul(dx, widths) + ctr_x
        local pred_ctr_y = torch.cmul(dy, heights) + ctr_y
        local pred_w = torch.cmul(torch.exp(dw), widths)
        local pred_h = torch.cmul(torch.exp(dh), heights)
        pred_boxes[{{}, {i}}] = pred_ctr_x - pred_w * 0.5  -- x1
        pred_boxes[{{}, {i+1}}] = pred_ctr_y - pred_h * 0.5 -- y1
        pred_boxes[{{}, {i+2}}] = pred_ctr_x + pred_w * 0.5 -- x2
        pred_boxes[{{}, {i+3}}] = pred_ctr_y +  pred_h * 0.5 -- y2

    end

    return pred_boxes
end

local function _clip_boxes(boxes, im_shape)
    -- Clip boxes to image boundaries.
    for i = 1, 84, 4 do
        boxes[{{}, {i}}]:cmax(0)  -- x1 >= 0
        boxes[{{}, {i+1}}]:cmax(0) -- y1 >= 0
        boxes[{{}, {i+2}}]:cmin(im_shape[3]-1)  -- x2 < width
        boxes[{{}, {i+3}}]:cmin(im_shape[2]-1)  -- y2 < height
    end
    return boxes

end

function unormalize(box_deltas, means, stds)
    -- Not used
    for i = 1, 84, 4 do
        local r = i / 4 + 1
        box_deltas[{{}, {i}}]:mul(stds[r][1]):add(means[r][1])
        box_deltas[{{}, {i+1}}]:mul(stds[r][2]):add(means[r][2])
        box_deltas[{{}, {i+2}}]:mul(stds[r][3]):add(means[r][3])
        box_deltas[{{}, {i+3}}]:mul(stds[r][4]):add(means[r][4])
    end
    return box_deltas
end


function im_detect(im, boxes, conv_ROI, cls_reg, means, stds)
    -- Detect object classes in an image given object proposals.
    -- Returns:
    --     scores: R x K array of object class scores
    --     boxes: R x 4 array of predicted bounding boxes
    
    -- boxes = boxes:double()
    boxes = boxes:add(-1):double()   -- needed when not perform add_regressiong...
    local blobs, unused_im_scale_factors = _get_blobs(im:clone(), boxes:clone())
    
    -- When mapping from image ROIs to feature map ROIs, there's some aliasing (some distinct image ROIs get mapped to the same feature ROI). Here we identify duplicate feature ROIs, so we only compute features on the unique subset.
    config.DEDUP_BOXES = 0
    if config.DEDUP_BOXES > 0 then
        print('dedup boxes...')
    end

    local conv_features_test = conv_ROI:forward(blobs)
    local blobs_out = cls_reg:forward( conv_features_test )
    local softmax = nn.Sequential()
    softmax:add(nn.SoftMax())
    local scores = softmax:forward(blobs_out[1])
    local box_deltas = blobs_out[2]

    local pred_boxes = _bbox_pred(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im:size())
    --[[
    for i = 1, scores:size(1) do
        be_print(pred_boxes[i], tostring(i))
        local mxs, idx = scores[i]:max(1)
        print('max scores: ', mxs, idx)
    end
    ]]
    return scores, pred_boxes
end

function apply_nms(all_boxes, thresh)
    -- Apply non-maximum suppression to all predicted boxes output by the test_net method
    local num_classes = #all_boxes
    local num_images = #all_boxes[1]
    local nms_boxes = {}
    for i = 1, num_classes do
        nms_boxes[i] = {}
    end
    for cls_ind = 1, num_classes do
        for im_ind = 1, num_images do
            nms_boxes[cls_ind][im_ind] = torch.Tensor(0)
            local dets = all_boxes[cls_ind][im_ind]
            
            if dets:dim() > 0 and dets[1][1] ~= -1 then
                local keep = utils.nms(dets, thresh)
                if keep:size(1) > 0 then
                    nms_boxes[cls_ind][im_ind] = dets:index(1, keep)
                end
            end
        end
    end
    return nms_boxes
end

local function _write_voc_results_file(all_boxes)
    local comp_id = 'torch7' 
    local path = './data/VOCdevkit/results/VOC2007/Main/' .. comp_id .. '_'
    for cls_ind, cls in pairs(cls_names) do
        if cls == '__background__' then
            break
        end
        print(string.format('Writing %s results file', cls))
        local filename = path .. 'det_test_' .. cls .. '.txt'
        print(filename)
        local wfile = io.open(filename, 'w')
        assert(wfile)
        -- image_index = {'000001', '000002'}  -- just for test
        for im_ind, index in pairs(images) do
            local dets = all_boxes[cls_ind][im_ind]
            if dets:dim() > 0 then
                for k = 1, dets:size(1) do
                    wfile:write(string.format('%s %.3f %.1f %.1f %.1f %.1f\n', index, dets[k][1], dets[k][2]+1, dets[k][3]+1, dets[k][4]+1, dets[k][5]+1))
                end
            end
        end
        wfile:close()
    end
    return comp_id
end

function _do_matlab_eval(comp_id, output_dir)
    local path = './VOCdevkit-matlab-wrapper/'
    local devkit_path = '../data/VOCdevkit/'
    output_dir = '.' .. output_dir
    local cmd = 'cd ' .. path .. ' && matlab -nodisplay -nodesktop'
    cmd = cmd .. ' -r "dbstop if error; '
    cmd = cmd .. string.format('voc_eval(\'%s\',\'%s\', \'%s\', \'%s\', %d); quit;"', devkit_path, comp_id, 'test', output_dir, 1)
    print('Running:\n' .. cmd)
    os.execute(cmd)
end

function evaluate_detections(nms_dets, output_dir)
    local comp_id = _write_voc_results_file(nms_dets)
    _do_matlab_eval(comp_id, output_dir)
end

local function _save_detections(all_boxes, output_dir)
    print('Saving all detected boxes to detections.h5 ...')
    local fn = output_dir .. 'detections.h5'
    local myFile = hdf5.open(fn, 'w')
    local num_classes = #all_boxes
    local num_images = #all_boxes[1]
    myFile:write('num_classes', torch.IntTensor({num_classes}))
    myFile:write('num_images', torch.IntTensor({num_images}))
    for cls_ind = 1, num_classes do
        for im_ind = 1, num_images do
            print(all_boxes[cls_ind][im_ind]:size())
            myFile:write(string.format('det_%d_%d', cls_ind, im_ind), all_boxes[cls_ind][im_ind])
        end
    end
    myFile:close()
    print('Done')
end

local function _load_detections(output_dir)
    print('Loading all detected boxes from detections.h5')
    local fn = output_dir .. 'detections.h5'
    local myFile = hdf5.open(fn, 'r')
    local num_classes = myFile:read('num_classes'):all()[1]
    local num_images = myFile:read('num_images'):all()[1]
    local all_boxes = {}
    for i = 1, num_classes do
        all_boxes[i] = {}
    end
    for cls_ind = 1, num_classes do
        for im_ind = 1, num_images do
            all_boxes[cls_ind][im_ind] = myFile:read(string.format('det_%d_%d', cls_ind, im_ind)):all()
        end
    end
    myFile:close()
    print('Done')
    return num_classes, num_images, all_boxes
end


function demo() 
    print('Demo: detecting image: ')
    -- images = pascal_voc.load_image_set_index()
    local gt_bboxes, gt_classes = pascal_voc.get_gt_bboxes()
    local ss_rois = pascal_voc.get_ss_rois()
    -- local means, stds, rois, bbox_targets, max_overlaps = 
        --ROIDetection.add_bbox_regression_targets( ss_rois, gt_bboxes, gt_classes ) 

    -- load model
    local conv_ROI, _ = get_model()
    local cls_reg = torch.load('trained_cls_reg.t7')
    conv_ROI:evaluate()
    cls_reg:evaluate()
    
    local im = pascal_voc.get_image('000014')
    scores, boxes = im_detect(im, ss_rois[10], conv_ROI, cls_reg)
    local CONF_THRESH = 0.7
    utils.visualize(im, boxes, scores, CONF_THRESH, cls_names)

end

function test_net()


    -- images = pascal_voc.load_image_set_index()
    local gt_bboxes, gt_classes = pascal_voc.get_gt_bboxes()
    local ss_rois = pascal_voc.get_ss_rois()
    --local means, stds, rois, bbox_targets, max_overlaps = 
        --ROIDetection.add_bbox_regression_targets( ss_rois, gt_bboxes, gt_classes ) 

    local conv_ROI, _ = get_model()
    local cls_reg = torch.load('trained_cls_reg.t7')
    conv_ROI:evaluate()
    cls_reg:evaluate()

    
    -- local num_images = #images
    local num_images = 2
    print(tostring(#images) .. ' images will be detected ...')
    -- heuristic: keep an average of 40 detections per class per images piror to NMS
    local max_per_set = 40 * num_images
    -- heuristic: keep at most 100 detections per class per iamge piror to NMS
    local max_per_image = 100
    -- detection thresold for each class (this is adaptively set based on the max_per_set constraint))
    local thresh = torch.ones(config.num_classes) * -math.huge
    be_print(thresh, 'thresh')

    --top_scores will hold one miniheap of scores per class (used to enforce the max_per_set constraint)
    -- all detections are collected into 
    --     all_boxes[cls][image] = N x 5 array of detections in (x1, y1, x2, y2, score)
    local top_scores = {}
    local all_boxes = {}
    for i = 1, config.num_classes-1 do
        top_scores[i] = utils.pqueue()
        all_boxes[i] = {}
    end

    for i = 1, num_images do
        local st = os.time()
        local im = pascal_voc.get_image(images[i])
        local scores, boxes = im_detect(im, ss_rois[i], conv_ROI, cls_reg)
        for j = 1, config.num_classes-1 do
            repeat
                local inds = scores[{{}, {j}}]:gt(thresh[j])
                local num_select = inds:sum()
                print('num_select: ', num_select)
                if num_select == 0 then
                    all_boxes[j][i] = torch.Tensor(1, 5):fill(-1)
                    print(all_boxes[j][i])
                    break
                end
                local cls_scores = scores[{{}, {j}}][inds]:reshape(num_select, 1)
                -- print(cls_scores)
                local inds_box = inds:cat(inds):cat(inds):cat(inds)
                local cls_boxes = boxes[{{}, {j*4-3, j*4}}][inds_box]:reshape(num_select, 4)
                cls_scores, inds = torch.sort(cls_scores, 1, true)
                local max_this_image = math.min(max_per_image, inds:size(1))
                cls_scores = cls_scores[{{1, max_this_image}, {}}]
                cls_boxes = cls_boxes:index(1, inds[{{1, max_this_image}, {}}]:reshape(max_this_image))

                -- push new scores onto the minheap
                for k = 1, max_this_image do
                    top_scores[j]:push(cls_scores[k][1])
                end
                -- if we've collected more than the max number of detection, 
                -- then pop items off the minheap and update the class threshold
                if table.getn(top_scores[j]) > max_per_set then
                    while table.getn(top_scores[j]) > max_per_set do
                        top_scores[j]:pop()
                    end
                    thresh[j] = top_scores[j][1]
                end

                all_boxes[j][i] = torch.cat(cls_scores, cls_boxes, 2)

                if false then
                    -- visualize
                end
            until true
        end
        be_print(thresh, 'thresh')
        print(string.format('The ' .. tostring(i) .. 'th image: ' .. images[i] .. '.jpg detected, elapsed time: %.2f', os.time()-st))
    end

    for j = 1, config.num_classes - 1 do
        for i = 1, num_images do
            local inds = all_boxes[j][i][{{}, {1}}]:gt(thresh[j])
            local num_keep = inds:sum()
            if num_keep == 0 then
                all_boxes[j][i] = torch.Tensor(1, 5):fill(-1)
            else 
                inds = inds:cat(inds):cat(inds):cat(inds):cat(inds)
                all_boxes[j][i] = all_boxes[j][i][inds]:reshape(num_keep, 5)
            end
        end
    end

    -- save to detections.h5
    local output_dir = './output/VGG_CNN_M_1024_7000/'
    _save_detections(all_boxes, output_dir)

    print('Applying NMS to all detections')
    local nms_dets = apply_nms(all_boxes, config.TEST_NMS) -- 0.3

    print('Evaluating detections')
    evaluate_detections(nms_dets, output_dir)

end

-- test_net()

local output_dir = './output/VGG_CNN_M_1024_7000/'
local fn = output_dir .. 'detections.h5'
if utils.file_exists(fn) then
    local num_classes, num_images, all_boxes = _load_detections(output_dir)
    --[[
    local im = pascal_voc.get_image(images[1])
    local CONF_THRESH = 0.5
    for cls = 1, 20 do
        print(all_boxes[cls][1][{{}, {2, 5}}])
        utils.visualize(im, all_boxes[cls][1][{{}, {2, 5}}], all_boxes[cls][1][{{}, {1}}], CONF_THRESH, cls_names)
    end
    ]]
    
    print('Applying NMS to all detections')
    local nms_dets = apply_nms(all_boxes, config.TEST_NMS) -- 0.3
    print('Evaluating detections')
    evaluate_detections(nms_dets, output_dir)

else
    test_net()
end

