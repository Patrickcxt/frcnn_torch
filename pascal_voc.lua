
pascal_voc = {}

require 'torch'
require 'image'
require 'hdf5'
require 'LuaXML'
require 'config'
require 'utils'

local num_classes = 21
local classes = {'__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor'}

local class_index = {}
for i = 1, num_classes do
    class_index[classes[i]] = i - 1
end
file_exists = utils.file_exists

function pascal_voc.get_image(index, year)
    year = year or config.year
    local path = './data/VOCdevkit/VOC' .. year .. '/JPEGImages/' .. tostring(index) .. '.jpg'
    local im = image.loadJPG(path)
    return im
end

function pascal_voc.get_ss_rois(year, data_type)
    -- Get seleselective search data 
    year = year or config.year
    data_type = data_type or config.data_type
    local path = './data/cache/'
    local target_file = path .. 'ss_voc_' .. year .. '_' .. data_type .. '.h5'
    if not file_exists(target_file) then
        -- print('ss roidb loaded from matlab files and creating ' .. target_file .. '..')
        local cmd = 'python mat_to_hdf5.py ' ..  year ..  ' ' .. data_type 
        os.execute(cmd)
    end
    print('ss roidb loaded from ' .. target_file .. '..')
    local myFile = hdf5.open(target_file)
    local num_images = myFile:read('num_images'):all()[1]
    ss_rois = {}
    for i = 1, num_images do
        local boxes = myFile:read(tostring(i-1)):all():float()
        table.insert(ss_rois, boxes)
    end
    myFile:close()
    print('Done')
    return ss_rois

end

local function _read_from_h5(path)
    local myFile = hdf5.open(path, 'r')
    num_images = myFile:read('num_images'):all()[1]
    gt_bboxes = {}
    gt_classes = {}
    for i = 1, num_images do
        table.insert(gt_bboxes, myFile:read('gt_bboxes_' .. tostring(i)):all())
    end
    for i = 1, num_images do
        table.insert(gt_classes, myFile:read('gt_classes_' .. tostring(i)):all())
    end
    myFile:close()
    return num_images, gt_bboxes, gt_classes
end

local function _write_to_h5(path, boxes, classes, num)
    local myFile = hdf5.open(path, 'w')
    local num_tensor = torch.IntTensor({num})
    myFile:write('num_images', num_tensor)
    for i = 1, num do
        myFile:write('gt_bboxes_' .. tostring(i), boxes[i])
    end
    for i = 1, num do
        myFile:write('gt_classes_' .. tostring(i), classes[i])
    end
    myFile:close()
end 

local function _load_pascal_annotation(year, index)
    local filename = './data/VOCdevkit/VOC' .. year .. '/Annotations/' .. tostring(index) .. '.xml'
    local file = xml.load(filename)
    local gt_classes = {}
    local gt_bbox = {}
    for _,item in pairs(file) do
        if item[item.TAG] == 'object' then
            local name = item:find('name')[1]
            table.insert(gt_classes, class_index[name])
            local bndbox = item:find('bndbox')
            local xmin = tonumber(bndbox:find('xmin')[1])
            local xmax = tonumber(bndbox:find('xmax')[1])
            local ymin = tonumber(bndbox:find('ymin')[1])
            local ymax = tonumber(bndbox:find('ymax')[1])
            table.insert(gt_bbox, {xmin, ymin, xmax, ymax})
        end
    end
    return torch.ShortTensor(gt_bbox), torch.ShortTensor(gt_classes)
end

function pascal_voc.get_gt_bboxes(year, data_type)
    -- Get gt roidb and classes

    year = year or config.year
    data_type = data_type or config.data_type
    cache_file = './data/cache/gt_voc_' .. year .. '_' .. data_type .. '.h5'
    local gt_bboxes = {}
    local gt_classes = {}
    local num_images = 0
    if file_exists(cache_file) then
        print('gt roidb loaded from ' .. cache_file .. '..')
        num_images, gt_bboxes, gt_classes = _read_from_h5(cache_file) 
    else
        print('gt roidb loaded from annotation xmls and creating ' .. cache_file .. '..')
        local image_index = pascal_voc.load_image_set_index(year, data_type)
        num_images = #image_index
        for im_i = 1, num_images do
            gt_bbox, gt_label = _load_pascal_annotation(year, image_index[im_i])
            table.insert(gt_bboxes, gt_bbox)
            table.insert(gt_classes, gt_label)
        end
        _write_to_h5(cache_file, gt_bboxes, gt_classes, num_images)
    end
    print('Done')
    return gt_bboxes, gt_classes
end


function pascal_voc.load_image_set_index(year, data_type)
    -- load the indexes listed in this dataset's image set file.
    -- Example path to image set file:
    -- ./data/VOCdevkit2007/VOC2007/ImageSets/Main/train.
    -- Note: only voc2007 data supported now
    year = year or config.year
    data_type = data_type or config.data_type
    local image_set_file = './data/VOCdevkit/VOC' .. year .. '/ImageSets/Main/' .. data_type .. '.txt'
    local file = io.open(image_set_file, 'r')
    local image_index = {}
    for line in file:lines() do
        table.insert(image_index, line)
    end
    return image_index
end

return pascal_voc
--gt_boxes, gt_classes = get_gt_bboxes('2007', 'train')
--local x = 5
--print(gt_boxes[x])
--print(gt_classes[x])
--image_index = _load_image_set_index('trainval')
--print(image_index)
--gt_bbox, gt_classes = _load_pascal_annotation('000001')
--print(type(gt_bbox[1]))
--ss_rois = get_ss_rois('2007', 'test')
