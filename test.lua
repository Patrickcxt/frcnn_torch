
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


function my_print(tb, name)
    print('=========================== ' .. name .. ' ===================================')
    io.read()
    for i = 1, 2 do
        print(tb[i])
        io.read()
    end
end

function _get_image_blob(im)
    -- Converts an image into a network input
    -- Return:
    --     blob: a data blob holding an image pyramid
    --     im_scale_factors(list): list of image scales (relative to im) used
    im, im_scales =  data_layer.prep_im_for_blob(im, config.PIXEL_MEANS, config.TEST_SCALES[1], config.TEST_MAX_SIZE)
    local im_blob = data_layer.im_list_to_blob({im})
    return im_blob, {im_scales}
end

function _get_rois_blob(im_rois, im_scales_factors)
    -- Convert RoIs into network inputs.
    -- Returns:
    --     blob: R x 5 matrix of RoIs in the image pyramid
    local rois  = data_layer.project_im_rois(im_rois, im_scales_factors[1])
    local rois_blob = torch.zeros(rois:size(1), 5)
    rois_blob[{{}, {1}}]:fill(1)
    rois_blob[{{}, {2, 5}}] = rois
    return rois_blob
end


function _get_blobs(im, im_rois)
    -- Convert an image and RoIs within that iamge into network inputs.
    local data, im_scale_factors = _get_image_blob(im)
    local rois = _get_rois_blob(im_rois, im_scale_factors)
    local input_blob = {data, rois}
    return input_blob, im_scale_factors
end

function _bbox_pred(boxes, box_deltas)
    -- Transform the set of class-agnostic boxes into class-specific boxes by applying the predicted offsets (box_deltas))
    box_deltas:double()
    my_print(boxes, 'boxes')
    if boxes:size(1) == 0 then
        return torch.zeros(0, box_deltas:size(2))
    end
    local widths = boxes[{{}, {3}}] - boxes[{{}, {1}}] + config._EPS
    my_print(widths, 'widths')
    local heights = boxes[{{}, {4}}] - boxes[{{}, {2}}] + config._EPS
    my_print(heights, 'heights')
    local ctr_x = boxes[{{}, {1}}] + widths * 0.5
    my_print(ctr_x, 'ctr_x')
    local ctr_y = boxes[{{}, {2}}] + heights * 0.5
    my_print(ctr_y, 'ctr_y')

    local pred_boxes = torch.zeros(box_deltas:size())
    for i = 1, 84, 4 do
        local dx = box_deltas[{{}, {i}}]
        local dy = box_deltas[{{}, {i+1}}]
        local dw = box_deltas[{{}, {i+2}}]
        local dh = box_deltas[{{}, {i+3}}]
        my_print(dx, 'dx')
        my_print(dy, 'dy')
        my_print(dw, 'dw')
        my_print(dh, 'dh')


        local pred_ctr_x = torch.cmul(dx, widths) + ctr_x
        local pred_ctr_y = torch.cmul(dy, heights) + ctr_y
        local pred_w = torch.cmul(torch.exp(dw), widths)
        local pred_h = torch.cmul(torch.exp(dh), heights)
        my_print(pred_ctr_x, 'pred_ctr_x')
        my_print(pred_ctr_y, 'pred_ctr_y')
        my_print(pred_w, 'pred_w')
        my_print(pred_h, 'pred_h')
        pred_boxes[{{}, {i}}] = pred_ctr_x - pred_w * 0.5  -- x1
        pred_boxes[{{}, {i+1}}] = pred_ctr_y - pred_h * 0.5 -- y1
        pred_boxes[{{}, {i+2}}] = pred_ctr_x + pred_w * 0.5 -- x2
        pred_boxes[{{}, {i+3}}] = pred_ctr_y +  pred_h * 0.5 -- y2
        my_print(pred_boxes[{{}, {i}}], 'x1')
        my_print(pred_boxes[{{}, {i+1}}], 'y1')
        my_print(pred_boxes[{{}, {i+2}}], 'x2')
        my_print(pred_boxes[{{}, {i+3}}], 'y2')
    end
    io.read()

    return pred_boxes
end

function _clip_boxes(boxes, im_shape)
    -- Clip boxes to image boundaries.
    for i = 1, 84, 4 do
        boxes[{{}, {i}}]:cmax(0)  -- x1 >= 0
        boxes[{{}, {i+1}}]:cmax(0) -- y1 >= 0
        boxes[{{}, {i+2}}]:cmin(im_shape:size(3)-1)  -- x2 < width
        boxes[{{}, {i+3}}]:cmin(im_shape:size(2)-1)  -- y2 < height
    end
    return boxes

end

function im_detect(im, boxes)
    -- Detect object classes in an image given object proposals.
    -- Returns:
    --     scores: R x K array of object class scores
    --     boxes: R x 4 array of predicted bounding boxes
    
    boxes = boxes:add(-1):double()
    local blobs, unused_im_scale_factors = _get_blobs(im:clone(), boxes:clone())
    print(unused_im_scale_factors)
    print(blobs[1]:size())
    print(im:size())
    --io.read()
    
    -- When mapping from image ROIs to feature map ROIs, there's some aliasing (some distinct image ROIs get mapped to the same feature ROI). Here we identify duplicate feature ROIs, so we only compute features on the unique subset.
    config.DEDUP_BOXES = 0
    if config.DEDUP_BOXES > 0 then
        print('dedup boxes...')
    end

    local conv_ROI, _ = get_model()
    local cls_reg = torch.load('trained_cls_reg.t7')
    conv_ROI:evaluate()
    cls_reg:evaluate()
    local conv_features_test = conv_ROI:forward(blobs)
    local blobs_out = cls_reg:forward( conv_features_test )
    local softmax = nn.Sequential()
    softmax:add(nn.SoftMax())
    local scores = softmax:forward(blobs_out[1])
    local box_deltas = blobs_out[2]
    --[[
    print('================ scores =============================')
    io.read()
    for i = 1, 10 do
        print(scores[i])
        io.read()
    end
    ]]
    my_print(box_deltas, 'box_deltas')

    local pred_boxes = _bbox_pred(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im:size())
    --my_print(pred_boxes, 'pred_boxes')
    --io.read()
    return scores, pred_boxes
end

function test_net()


    local cls_names = {
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 
        'cow', 'diningtable', 'dog', 'horse',
        'motobike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor',
        '__background__'
    }
    local images = pascal_voc.load_image_set_index()
    local gt_bboxes, gt_classes = pascal_voc.get_gt_bboxes()
    local ss_rois = pascal_voc.get_ss_rois()
    -- local means, stds, rois, bbox_targets, max_overlaps = 
    --    ROIDetection.add_bbox_regression_targets( ss_rois, gt_bboxes, gt_classes ) 

    local im = pascal_voc.get_image('000004')
    scores, boxes = im_detect(im, ss_rois[4])
    local CONF_THRESH = 0.8
    --print(scores)
    --io.read()
    --print(boxes)
    --io.read()
    utils.visualize(im, boxes, scores, CONF_THRESH, cls_names)
    --[[
    local NMS_THRESH = 0.3
    for i = 1, num_images do
        local im = pascal_voc.get_image(images[i])
        local scores, boxes = im_dect(im, ss_rois[i])
        utils.visualize(im, boxes, scores, CONF_THRESH, cls_names)
    end
    ]]

end

test_net()

