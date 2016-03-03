import os
import numpy as np
import scipy.io as sio
import h5py
import sys

def load_selective_search_roidb(year, data_type):
    path = './data/selective_search_data/'
    target_path = './data/cache/'
    filename = path + 'voc_' + year \
            + '_' + data_type + '.mat'
    print('loading selective search rois from ' + filename + '...')
    assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
    raw_data = sio.loadmat(filename)['boxes'].ravel()
    f = h5py.File(target_path+'ss_voc_'+year+'_'+data_type+'.h5', 'w')
    box_list = []
    for i in xrange(raw_data.shape[0]):
        boxes = raw_data[i][:, (1, 0, 3, 2)]
        box_list.append(boxes)
        #print(len(raw_data[i][:, (1, 0, 3, 2)]))
        f.create_dataset(str(i), data = boxes)
    num_images = np.ndarray((1), dtype=int)
    num_images[0] = raw_data.shape[0]
    f.create_dataset('num_images', data = num_images)
    f.close()
    return box_list

if __name__ == '__main__':
    year = sys.argv[1]
    data_type = sys.argv[2]
    box_list = load_selective_search_roidb(year, data_type)
    print('done...')

