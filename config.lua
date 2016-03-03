
config = {}
-- source of training data
config.year = '2007'
config.data_type = 'train'

config.num_classes = 21
config._EPS = 1e-14
config.TRAIN_BBOX_THRESH = 0.5

-- Pixel mean values (BGR order) as a (1, 1, 3) array
-- We use the same pixel mean for all networks even though it's not exactly what
-- they were trained with
config.PIXEL_MEANS = {102.9801, 115.9465, 122.7717}
return config
