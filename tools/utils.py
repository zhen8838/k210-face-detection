import tensorflow as tf
import numpy as np
import os
import skimage
import cv2
from math import cos, sin
from imgaug import augmenters as iaa
import imgaug as ia


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
        self.iaaseq = iaa.Sequential([
            iaa.Fliplr(0.5),  # 50% 镜像
            iaa.Crop(percent=(0, 0.1)),  # random crops
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.5, 1.5)),
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                       rotate=(-10, 10))
        ])

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

    def data_augmenter(self, img, true_box):
        seq_det = self.iaaseq.to_deterministic()
        img = img.astype('uint8')

        bbs = ia.BoundingBoxesOnImage.from_xyxy_array(self.center_to_corner(true_box), shape=(self.in_h, self.in_w))

        image_aug = seq_det.augment_images([img])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

        xyxy_box = bbs_aug.to_xyxy_array()
        new_box = self.corner_to_center(xyxy_box)
        return image_aug, new_box

    def generator(self, is_training=True, is_resize=True, is_make_lable=True):
        with open(self.list_name, 'r') as f:
            datalist = f.readlines()
        self.total_data = len(datalist)
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
                for i in range(1, len(one_ann), 5):
                    true_box.append(one_ann[i:i+5])
                true_box = np.asfarray(true_box)
                # todo data augment
                if is_resize:
                    img = skimage.transform.resize(img, (self.in_h, self.in_w), mode='reflect', preserve_range=True)

                if is_training:
                    img, true_box = self.data_augmenter(img, true_box)

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
        self.epoch_step = self.total_data//batch_size

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

    def center_to_corner(self, true_box):
        x1 = (true_box[:, 0:1]-true_box[:, 2:3])*self.in_w
        y1 = (true_box[:, 1:2]-true_box[:, 3:4])*self.in_h
        x2 = (true_box[:, 0:1]+true_box[:, 2:3])*self.in_w
        y2 = (true_box[:, 1:2]+true_box[:, 3:4])*self.in_h
        xyxy_box = np.hstack([x1, y1, x2, y2])
        return xyxy_box.astype('float32')

    def corner_to_center(self, xyxy_box):
        x = ((xyxy_box[:, 2:3]-xyxy_box[:, 0:1])/2+xyxy_box[:, 0:1])/self.in_w
        y = ((xyxy_box[:, 3:4]-xyxy_box[:, 1:2])/2+xyxy_box[:, 1:2])/self.in_h
        w = (xyxy_box[:, 2:3]-xyxy_box[:, 0:1])/(2*self.in_w)
        h = (xyxy_box[:, 3:4]-xyxy_box[:, 1:2])/(2*self.in_h)
        true_box = np.hstack([x, y, w, h])
        return true_box.astype('float32')
