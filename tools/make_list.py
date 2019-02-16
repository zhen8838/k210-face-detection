import sys
import os
import skimage
import numpy as np


def create_list(fddb_dir, ann_dir):
    datafiles = [name for name in os.listdir(ann_dir) if 'ellipseList' in name]
    datafiles.sort()
    datalist = []
    for name in datafiles:
        with open(os.path.join(ann_dir, name)) as f:
            datalist.extend(f.readlines())
    datalist = [line.strip()+' ' for line in datalist]
    for i, line in enumerate(datalist):
        if 'img' in line:
            datalist[i] = '\n'+os.path.join(fddb_dir, line.strip())+'.jpg'
            datalist[i+1] = ' '
    datalist[0] = datalist[0].lstrip()
    with open('data/train.list', 'w') as f:
        f.writelines(datalist)


def main(fddb_dir='/home/zqh/FDDB', ann_dir='/home/zqh/FDDB/FDDB-folds'):
    create_list(fddb_dir, ann_dir)
    """ rescale the xywh """
    new_datalist = []
    with open('data/train.list', 'r') as f:
        datalist = f.readlines()

    with open('data/train.list', 'w') as f:
        for line in datalist:
            one_ann = line.strip().split()
            img_path = one_ann[0]
            img = skimage.io.imread(img_path)
            true_box = []
            for i in range(1, len(one_ann), 6):
                true_box.append(one_ann[i:i+6])
            true_box = np.asfarray(true_box, dtype='float32')
            # NOTE convert the [h w] to [w h]
            true_box[:, [0, 1]] = true_box[:, [1, 0]]
            # NOTE convert [w,h,ang,x,y,1] to [x,y,w,h,1]
            true_box = true_box[:, [3, 4, 0, 1, 5]]
            # convert xy wh to [0-1]
            true_box[:, 0:2] /= img.shape[0:2][::-1]
            true_box[:, 2:4] /= img.shape[0:2][::-1]
            # true_box[:, 2:4] *= 2 # NOTE the fddb annotation is radius
            true_box = true_box.astype('float32')
            f.write(img_path+' ')
            for box in true_box:
                np.savetxt(f, box, fmt='%f', newline=' ')
            f.write('\n')


if __name__ == "__main__":
    main()
