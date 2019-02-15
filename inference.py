from models.yolonet import yoloconv
import tensorflow as tf
from tools.utils import helper
import skimage
import numpy as np
from scipy.special import expit
import sys
import argparse


def main(pb_path, image_size):
    g = tf.get_default_graph()
    fddb = helper('data/train.list', image_size, (7, 10))
    test_img = fddb._read_img('data/2.jpeg', True)
    test_img = fddb._process_img(test_img, None, is_training=False)[0]

    with tf.gfile.GFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    inputs = g.get_tensor_by_name('Input_image:0')
    pred_label = g.get_tensor_by_name('Yolo/Final/conv2d/BiasAdd:0')

    with tf.Session() as sess:
        test_img = test_img[np.newaxis, :, :, :]
        pred_label_ = sess.run(pred_label, feed_dict={inputs: test_img})

        pred_label_ = expit(pred_label_)
        boxes = fddb.label_to_box(pred_label_[0])
        fddb.draw_box(test_img[0], boxes)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--pb_path', type=str, help='pb file path', default='Freeze_save.pb')
    parser.add_argument('--image_size', type=int, help='net work input image size', default=(240, 320), nargs='+')
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args.pb_path, args.image_size)
