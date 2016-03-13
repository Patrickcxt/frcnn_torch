--------------------------------------------------------------------------------
-- utility functions for the evaluation part
--------------------------------------------------------------------------------


utils = {}

function utils.boxoverlap(a,b)
  local b = b.xmin and {b.xmin,b.ymin,b.xmax,b.ymax} or b
    
  local x1 = a:select(2,1):clone()
  x1[x1:lt(b[1])] = b[1] 
  local y1 = a:select(2,2):clone()
  y1[y1:lt(b[2])] = b[2]
  local x2 = a:select(2,3):clone()
  x2[x2:gt(b[3])] = b[3]
  local y2 = a:select(2,4):clone()
  y2[y2:gt(b[4])] = b[4]
  
  local w = x2-x1+1;
  local h = y2-y1+1;
  local inter = torch.cmul(w,h):float()
  local aarea = torch.cmul((a:select(2,3)-a:select(2,1)+1) ,
                           (a:select(2,4)-a:select(2,2)+1)):float()
  local barea = (b[3]-b[1]+1) * (b[4]-b[2]+1);
  
  -- intersection over union overlap
  local o = torch.cdiv(inter , (aarea+barea-inter))
  -- set invalid entries to 0 overlap
  o[w:lt(0)] = 0
  o[h:lt(0)] = 0
  
  return o
end


-------------------------------------------------------------------------------
-- Whether file exists
-------------------------------------------------------------------------------
function utils.file_exists(name)
    local f = io.open(name, 'r')
    if f ~= nil then
        io.close(f)
        return true
    else
        return false;
    end
end

-------------------------------------------------------------------------------
-- Non-maximum suppression
-------------------------------------------------------------------------------
function utils.nms(boxes, overlap)
  
  local pick = torch.LongTensor()

  if boxes:numel() == 0 then
    return pick
  end

  local x1 = boxes[{{},2}]
  local y1 = boxes[{{},3}]
  local x2 = boxes[{{},4}]
  local y2 = boxes[{{},5}]
  local s = boxes[{{},1}]
  
  local area = boxes.new():resizeAs(s):zero()
  area:map2(x2,x1,function(xx,xx2,xx1) return xx2-xx1+1 end)
  area:map2(y2,y1,function(xx,xx2,xx1) return xx*(xx2-xx1+1) end)

  local vals, I = s:sort(1)

  pick:resize(s:size()):zero()
  local counter = 1
  local xx1 = boxes.new()
  local yy1 = boxes.new()
  local xx2 = boxes.new()
  local yy2 = boxes.new()

  local w = boxes.new()
  local h = boxes.new()

  while I:numel()>0 do 
    local last = I:size(1)
    local i = I[last]
    pick[counter] = i
    counter = counter + 1
    if last == 1 then
      break
    end
    I = I[{{1,last-1}}]
    
    xx1:index(x1,1,I)
    xx1:cmax(x1[i])
    yy1:index(y1,1,I)
    yy1:cmax(y1[i])
    xx2:index(x2,1,I)
    xx2:cmin(x2[i])
    yy2:index(y2,1,I)
    yy2:cmin(y2[i])
    
    w:resizeAs(xx2):zero()
    w:map2(xx2,xx1,function(xx,xxx2,xxx1) return math.max(xxx2-xxx1+1,0) end)
    h:resizeAs(yy2):zero()
    h:map2(yy2,yy1,function(xx,yyy2,yyy1) return math.max(yyy2-yyy1+1,0) end)
    
    local inter = w
    inter:cmul(h)

    local o = h
    xx1:index(area,1,I)
    torch.cdiv(o,inter,xx1+area[i]-inter)
    I = I[o:le(overlap)]
  end

  pick = pick[{{1,counter-1}}]
  return pick
end

-------------------------------------------------------------------
--  visualize
-------------------------------------------------------------------
function utils.visualize(im, boxes, scores, thresh, cl_names)
  
  local ok = pcall(require,'qt')
  if not ok then
    error('You need to run visualize_detections using qlua')
  end
  require 'qttorch'
  require 'qtwidget'

  -- select best scoring boxes without background
  local max_score,idx = scores[{{},{1, 20}}]:max(2)

  local idx_thresh = max_score:gt(thresh)
  max_score = max_score[idx_thresh]
  idx = idx[idx_thresh]

  local r = torch.range(1,boxes:size(1)):long()
  local rr = r[idx_thresh]
  if rr:numel() == 0 then
    error('No detections with a score greater than the specified threshold')
  end
  local boxes_thresh = boxes:index(1,rr)
  
  local keep = utils.nms(torch.cat(max_score:float(), boxes_thresh:float(),2),0.3)
  
  boxes_thresh = boxes_thresh:index(1,keep)
  max_score = max_score:index(1,keep)
  idx = idx:index(1,keep)

  local num_boxes = boxes_thresh:size(1)
  local widths  = boxes_thresh[{{},3}] - boxes_thresh[{{},1}]
  local heights = boxes_thresh[{{},4}] - boxes_thresh[{{},2}]

  local x,y = im:size(3),im:size(2)
  local w = qtwidget.newwindow(x,y,"Detections")
  local qtimg = qt.QImage.fromTensor(im)
  w:image(0,0,x,y,qtimg)
  local fontsize = 15

  for i=1,num_boxes do
    local x,y = boxes_thresh[{i,1}],boxes_thresh[{i,2}]
    local width,height = widths[i], heights[i]
    
    -- add bbox
    w:rectangle(x,y,width,height)
    
    -- add score
    w:moveto(x,y+fontsize)
    w:setcolor("red")
    w:setfont(qt.QFont{serif=true,italic=true,size=fontsize,bold=true})
    if cl_names then
      w:show(string.format('%s: %.2f',cl_names[idx[i]],max_score[i]))
    else
      w:show(string.format('%d: %.2f',idx[i],max_score[i]))
    end
  end
  w:setcolor("red")
  w:setlinewidth(2)
  w:stroke()
  return w
end


return utils


