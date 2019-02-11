import tensorflow as tf
import numpy as np
import os
import skimage
import cv2
from math import cos, sin


def make_train_list():
    train_txt_path = '/media/zqh/Datas/DataSet/FDDB/FDDB-folds/train.txt'
    image_root = '/media/zqh/Datas/DataSet/FDDB/'
    with open(train_txt_path, 'r') as f:
        datalist = f.readlines()
    datalist = [i.strip() for i in datalist]
    for i, line in enumerate(datalist):
        if 'img' in line:
            datalist[i] = '\n'+image_root+line+'.jpg'
            datalist[i+1] = ''
        else:
            datalist[i] = ' '+line
    with open('data/train.list', 'w') as f:
        f.write(''.join(datalist))

def make_test_list():
    train_txt_path = '/media/zqh/Datas/DataSet/FDDB/FDDB-folds/test.txt'
    image_root = '/media/zqh/Datas/DataSet/FDDB/'
    with open(train_txt_path, 'r') as f:
        datalist = f.readlines()
    datalist = [i.strip() for i in datalist]
    for i, line in enumerate(datalist):
        if 'img' in line:
            datalist[i] = '\n'+image_root+line+'.jpg'
            datalist[i+1] = ''
        else:
            datalist[i] = ' '+line
    with open('data/test.list', 'w') as f:
        f.write(''.join(datalist))


def rotate_img(img, true_box):
    # 旋转图像,那么坐标轴也要交换
    """ w h an x y """

    true_box[:, [3, 4]] = true_box[:, [4, 3]]
    true_box[:, [0, 1]] = true_box[:, [1, 0]]
    return skimage.transform.rotate(img, -90., resize=True)


class helper(object):
    def __init__(self, list_name, in_hw: tuple, out_hw: tuple):
        self.in_h = in_hw[0]
        self.in_w = in_hw[1]
        self.out_h = out_hw[0]
        self.out_w = out_hw[1]
        self.list_name = list_name
        self.grid_w = 1/self.out_w
        self.grid_h = 1/self.out_h
        self.xy_offset = self._coordinate_offset()

    def _xy_to_grid(self, box: np.ndarray)->tuple:
        if box[0] == 1.0:
            idx, modx = self.out_w-1, 1.0
        else:
            idx, modx = divmod(box[0], self.grid_w)
            modx /= self.grid_w

        if box[1] == 1.0:
            idy, mody = self.out_h-1, 1.0
        else:
            idy, mody = divmod(box[1], self.grid_h)
            mody /= self.grid_h
        return int(idx), modx, int(idy), mody

    def box_to_label(self, img, true_box):
        label = np.zeros((self.out_h, self.out_w, 5))
        for box in true_box:
            idx, modx, idy, mody = self._xy_to_grid(box)
            label[idy, idx, 0] = modx  # x
            label[idy, idx, 1] = mody  # y
            label[idy, idx, 2] = box[2]  # w
            label[idy, idx, 3] = box[3]  # h
            label[idy, idx, 4] = 1
        return label

    def _coordinate_offset(self):
        offset = np.zeros((self.out_h, self.out_w, 2))
        for i in range(self.out_h):
            for j in range(self.out_w):
                offset[i, j, :] = np.array([j, i])  # NOTE  [x,y]
        offset[..., 0] /= self.out_w
        offset[..., 1] /= self.out_h
        return offset.astype('float32')

    def _xy_to_all(self, label: np.ndarray)->np.ndarray:
        label[:, :, 0:2] = label[:, :, 0:2] * np.array([self.grid_w, self.grid_h])+self.xy_offset

    def label_to_box(self, label):
        self._xy_to_all(label)
        true_box = label[np.where(label[:, :, 4] > .7)]
        return true_box.astype('float32')

    def generator(self, is_resize=True, is_make_lable=True):
        with open(self.list_name, 'r') as f:
            datalist = f.readlines()
        while True:
            for line in datalist:
                if line.strip() == '':
                    continue
                one_ann = line.strip().split()
                img_path = one_ann[0]
                img = skimage.io.imread(img_path)
                if len(img.shape) != 3:
                    img = skimage.color.gray2rgb(img)
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
                if is_resize:
                    img = skimage.transform.resize(img, (self.in_h, self.in_w), mode='reflect')
                # normalize image to [0-1]
                img = skimage.exposure.equalize_hist(img)
                if is_make_lable:
                    yield img, self.box_to_label(img, true_box)
                else:
                    yield img, true_box

    def set_dataset(self, batch_size, rand_seed):
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.float32),
                                                 (tf.TensorShape([self.in_h, self.in_w, 3]), tf.TensorShape([self.out_h, self.out_w, 5])))
        dataset = dataset.batch(batch_size, True)
        dataset = dataset.prefetch(batch_size)
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(batch_size*2, count=None, seed=rand_seed))
        self.dataset = dataset

    def get_iter(self):
        return self.dataset.make_one_shot_iterator().get_next()

    def draw_box(self, img, true_box):
        """ [x,y,w,h,1] """
        for box in true_box:
            cv2.rectangle(img, tuple(((box[0:2]-box[2:4])*img.shape[0:2][::-1]).astype('int')),
                          tuple(((box[0:2] + box[2:4])*img.shape[0:2][::-1]).astype('int')),
                          color=(0, 200, 0))
        skimage.io.imshow(img)
        skimage.io.show()
