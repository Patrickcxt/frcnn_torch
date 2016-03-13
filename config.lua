
config = {}
-- source of training data
config.year = '2007'
config.data_type = 'test'

config.num_classes = 21
config._EPS = 1e-14


config.TRAIN_BBOX_THRESH = 0.5

-- Scales to use during traing (can list multiple scales)
-- Each scale is the pixel size of an image's shortest side
config.TRAIN_SCALES = {600}

-- Images to user per minibatch
config.TRAIN_IMS_PER_BATCH = 2

-- Minibatch size (number of regions of interest[ROI]) 
config.TRAIN_BATCH_SIZE =  128

-- Fractionof minibatch that is labeled foreground (i.e. class > 0)
config.TRAIN_FG_FRACTION = 0.25

-- Overlap threshold for a RoI to be considered foreground (if >= FG_THRESH)
config.TRAIN_FG_THRESH = 0.5

-- Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
config.TRAIN_BG_THRESH_HI = 0.5
config.TRAIN_BG_THRESH_LO = 0.1

-- Max pixel size of the longest side of a scaled input image
config.TRAIN_MAX_SIZE = 1000


-- Pixel mean values (BGR order) as a (1, 1, 3) array
-- We use the same pixel mean for all networks even though it's not exactly what
-- they were trained with
config.PIXEL_MEANS = {122.7717, 115.9465, 102.9801}




config.TEST_SCALES = {600}
config.TEST_MAX_SIZE = 1000
config.DEDUP_BOXES = 1.0/16.0


return config
