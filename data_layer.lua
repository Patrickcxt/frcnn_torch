require 'torch'
require 'nn'
require 'image'
require 'config'
require 'utils'

data_layer = {}
local _roidb = {}
local _cur = 1
local _perm

local function _index(tb, st, ed)
    -- Return elements between index std and ed in the table
    local storage = {}
    for i = 0, ed-1 do
        table.insert(storage, tb[st+i])
    end
    return storage
end

local function _index_sp(tb, inds)
    -- Return tb elements in the inds. 
    -- inds = torch.totable(inds) 
    local storage = {}
    for i = 1, #inds do
        table.insert(storage, tb[inds[i]]) 
    end
    return storage
end

local function _index_imgsp( im1, im2 )
    -- Return elements in roidb whose 'image' field is specified by im1, im2
    -- This function used for testing.
    local storage = {}
    for i = 1, #_roidb  do
        if _roidb[i]['image'] == im1 then
            table.insert(storage, _roidb[i]) 
        end
    end
    for i = 1, #_roidb  do
        if _roidb[i]['image'] == im2 then
            table.insert(storage, _roidb[i]) 
        end
    end
    return storage
end

local function _shuffle_roidb_inds()
    -- Randomly permute the training roidb.
    -- print('>>>>>>>> _shuffle_roidb_inds')
    _perm = torch.randperm(#_roidb)
    _cur = 1
    -- print('<<<<<<<< _shuffle_roidb_inds')
end

local function _get_next_minibatch_inds()
    -- Return the roidb indices for the next minibatch.
    -- print('>>>>>>>> get_next_minibatch_inds')

    if _cur + config.TRAIN_IMS_PER_BATCH > #_roidb then
        _shuffle_roidb_inds()
    end
    local db_inds = _index(_perm, _cur, config.TRAIN_IMS_PER_BATCH)
    _cur = _cur + config.TRAIN_IMS_PER_BATCH
    --print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> _cur: ' .. tostring(_cur))
    -- print('<<<<<<< get_next_minibatch_inds')
    return db_inds
end


function data_layer.prep_im_for_blob(im, pixel_means, target_size, max_size)
    -- Mean subtract and scale an image for use in a blob
    im = im:mul(255.0):clamp(0, 255)
    local p = config.PIXEL_MEANS
    im[1]:add(-p[1])
    im[2]:add(-p[2])
    im[3]:add(-p[3])
    local im_size_min = math.min(im:size(2), im:size(3))
    local im_size_max = math.max(im:size(2), im:size(3))
    local im_scale = target_size / im_size_min
    -- Prevent the biggest axis from being more than MAX_SIZE
    if torch.round(im_scale * im_size_max) > max_size then
        im_scale = max_size / im_size_max
    end
    im = image.scale(im, im:size(3) * im_scale, im:size(2) * im_scale)
    return im , im_scale
end

function data_layer.im_list_to_blob(ims)
    -- print('>>>>>>>> _im_list_to_blob')
    -- Convert a list of images into a network input
    -- Assumes images are already prepared (means substracted ...)
    local shapes = torch.Tensor(#ims, 3)
    for im_i = 1, #ims do
        shapes[im_i] = torch.Tensor({ims[im_i]:size(1), ims[im_i]:size(2), ims[im_i]:size(3)})
    end
    -- print(shapes)
    local max_shape, _ = shapes:max(1)
    -- print(max_shape)
    local blob = torch.zeros(#ims, 3, max_shape[1][2], max_shape[1][3])
    for i = 1, #ims do
        blob[{{i}, {}, {1, ims[i]:size(2)}, {1, ims[i]:size(3)}}] = ims[i]
    end
    -- print('<<<<<<<< _im_list_to_blob')
    return blob
end

local function _get_image_blob(roidb, scale_inds)
    -- Builds an input blob from the images in the roidb at the specified scales
    -- print('>>>>>>>> _get_image_blob')
    local num_images = #roidb
    local processed_ims = {}
    local im_scales = {}
    for i = 1, num_images do
        print('image: ' .. roidb[i]['image'] .. '.jpg')
        local im = pascal_voc.get_image(roidb[i]['image'])
        local im_scale;
        -- To do: flipped
        local target_size = config.TRAIN_SCALES[scale_inds[i]]
        im, im_scale = data_layer.prep_im_for_blob(im, config.PIXEL_MEANS, target_size, config.TRAIN_MAX_SIZE)
        table.insert(im_scales, im_scale)
        table.insert(processed_ims, im)
    end
    local blob = data_layer.im_list_to_blob(processed_ims)
    -- print('<<<<<<<< _get_image_blob')
    return blob, im_scales

end

function data_layer.project_im_rois(im_rois, im_scales_factor)

    -- print('>>>>>>>> _project_im_rois')
    -- Project image RoIs into the rescaled training image.
    local rois = im_rois * im_scales_factor
    -- print('<<<<<<<< _project_im_rois')
    return rois
end

local function _get_bbox_regression_labels(bbox_target_data, clss)
    -- Bounding-box regression are stored in a compact form in the roidb
    -- This function expands those targets into the 4-of-4*k representation used by the network (i.e. only one class has non-zero target). 
    -- Returns:
    --   bbox_target_data: N x 4K blob of regression targets
    
    -- print('>>>>>>>> _get_bbox_regression_labels')
    local num = clss:size(1)
    local bbox_targets = torch.zeros(num, 4 * config.num_classes)
    for ind = 1, num do
        local cls = clss[ind][1]
        if cls < 21 then
            local st = 4 * (cls-1) + 1
            local ed = st + 3
            bbox_targets[{{ind}, {st, ed}}] = bbox_target_data[{{ind}, {2, 5}}]
        end
    end
    -- print('<<<<<<<< _get_bbox_regression_labels')
    return bbox_targets
end

local function _sample_rois(roidb, fg_rois_per_image, rois_per_image)
    -- Generate a random smaple of RoIs comprising foreground and background examples
    -- print('>>>>>>>> _sample_rois')
    local rois = roidb['boxes']
    local overlaps = roidb['max_overlaps']
    local labels = roidb['bbox_targets'][{{}, {1}}]
    local num_rois = rois:size(1)

    -- Select foreground RoIs as those with >= FG_THRESH overlap
    -- Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI]
    local fg_inds = {}
    local bg_inds = {}
    for i = 1, num_rois do
        if overlaps[i][1] >= config.TRAIN_FG_THRESH then
            table.insert(fg_inds, i)
        end
        if overlaps[i][1] <  config.TRAIN_BG_THRESH_HI and overlaps[i][1] >= config.TRAIN_BG_THRESH_LO then
            table.insert(bg_inds, i)
        end
    end
    local fg_rois_per_this_image = math.min(#fg_inds, fg_rois_per_image)
    if #fg_inds > 0 then
        fg_inds = _index_sp(fg_inds, torch.totable((torch.randperm(#fg_inds))[{{1, fg_rois_per_this_image}}]))
    end
    local bg_rois_per_this_image = math.min(rois_per_image - fg_rois_per_this_image, #bg_inds)
    if #bg_inds > 0 then
        bg_inds = _index_sp(bg_inds, torch.totable((torch.randperm(#bg_inds))[{{1, bg_rois_per_this_image}}]))
    end
    
    local rois_keep = torch.zeros(#fg_inds + #bg_inds, 4)
    local targets_keep = torch.zeros(#fg_inds + #bg_inds, 5)
    local labels_keep = torch.zeros(#fg_inds + #bg_inds, 1):fill(21)
    local j = 1
    for i = 1, #fg_inds do
        rois_keep[j] = rois[fg_inds[i]]
        targets_keep[j] = roidb['bbox_targets'][fg_inds[i]]
        labels_keep[j] = labels[fg_inds[i]]
        j = j + 1
    end
    for i = 1, #bg_inds do
        rois_keep[j] = rois[bg_inds[i]]
        targets_keep[j] = roidb['bbox_targets'][bg_inds[i]]
        j = j + 1
    end

    bbox_targets = _get_bbox_regression_labels(targets_keep, labels_keep)
    -- print('<<<<<<<< _sample_rois')
    return rois_keep, bbox_targets, labels_keep
end

local function _get_minibatch(roidb)
    -- Given a roidb, construct a minibatch sampled from it
    local num_images = #roidb
    print('num of images: ', num_images)
    -- Sample random scales to use for each image in this batch
    -- assert BATCH_SIZE % num_images == 0
    local rois_per_image = config.TRAIN_BATCH_SIZE / num_images
    local fg_rois_per_image = torch.round(config.TRAIN_FG_FRACTION * rois_per_image)
    -- print(rois_per_image, fg_rois_per_image)
    
    -- Get the input image blob
    local im_blob, im_scales = _get_image_blob(roidb, {1, 1})

    -- Now, build the region of interest and label blobs
    local rois_blob = torch.zeros(1, 5)
    local bbox_targets_blob = torch.zeros(1, 4 * config.num_classes)
    local labels_blob = torch.zeros(1, 1)
    -- bbox_loss_blob = torch.zeros(0, 4 * config.num_classes)
    for im_i = 1, num_images do
        local im_rois, bbox_targets, clss = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image)
        
        -- Add to RoIs blob
        local  rois = data_layer.project_im_rois(im_rois, im_scales[im_i])
        local  rois_blob_this_image = torch.zeros(im_rois:size(1), 5)
        rois_blob_this_image[{{}, {1} }]:fill(im_i)
        rois_blob_this_image[{{}, {2, 5} }] = rois
        rois_blob = torch.cat(rois_blob, rois_blob_this_image, 1)

        -- Add to labels, bbox targets, and bbox loss blobs
        bbox_targets_blob = torch.cat(bbox_targets_blob, bbox_targets, 1)
        labels_blob = torch.cat(labels_blob, clss, 1) 
        -- bbox_loss_blob = torch.cat(bbox_loss_blob, bbox_loss)
    end
    rois_blob = rois_blob[{{2, rois_blob:size(1)}, {}}]
    bbox_targets_blob = bbox_targets_blob[{{2, bbox_targets_blob:size(1)}, {}}]
    labels_blob = labels_blob[{{2, labels_blob:size(1)}, {}}]
    return im_blob, rois_blob,  bbox_targets_blob, labels_blob
end

function data_layer.get_next_minibatch()
    -- Return the blobs to be used for the next minibatch.
    print('Getting next minibatch ... ')
    local db_inds = _get_next_minibatch_inds()
    local minibatch_db = _index_sp(_roidb, db_inds)
    local im_blob, rois_blob, bbox_targets_blob, labels_blob =  _get_minibatch(minibatch_db)

    print('Done')
    return im_blob, rois_blob, bbox_targets_blob, labels_blob
end


function data_layer.set_roidb(images, rois, bbox_targets, max_overlaps)
    -- print('>>>>>>> set_roidb')
    local roidb = {}
    local num_images = #images
    for im_i = 1, num_images do
        local roidb_tmp = {}
        roidb_tmp['image'] = images[im_i]
        roidb_tmp['boxes'] = rois[im_i]
        roidb_tmp['bbox_targets'] = bbox_targets[im_i]
        roidb_tmp['max_overlaps'] = max_overlaps[im_i]
        table.insert(roidb, roidb_tmp)
    end
    _roidb = roidb
    _shuffle_roidb_inds()
    -- print('<<<<<<< set_roidb')
end

return data_layer
