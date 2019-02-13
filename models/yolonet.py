import tensorflow as tf
from tensorflow.contrib import slim
from models.mobilenet_v1 import *


def yolonet(images: tf.Tensor, depth_multiplier: float, is_training: bool):
    flower_point = ['Conv2d_0_depthwise', 'Conv2d_0_pointwise',  'Final']
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
        nets, endpoints = mobilenet_v1_base(images, depth_multiplier=depth_multiplier)

    # add the new layer
    with tf.variable_scope('Yolo'):
        with slim.arg_scope([slim.batch_norm], is_training=is_training,  center=True, scale=True, decay=0.9997, epsilon=0.001):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME', normalizer_fn=slim.batch_norm, activation_fn=None,):
                # (?, 8, 10, 512) ===> (?, 4, 5, 512)
                nets = slim.conv2d(nets, 256, (3, 3), scope=flower_point[0])
                nets = tf.nn.relu6(nets, name=flower_point[0]+'/relu6')
                endpoints[flower_point[0]] = nets
                # nets = (?, 8, 10, 128)
                nets = slim.conv2d(nets, 128, (3, 3), scope=flower_point[1])
                nets = tf.nn.relu6(nets, name=flower_point[1]+'/relu6')
                endpoints[flower_point[1]] = nets
                # nets = (?, 8, 10, 128)
                nets = slim.conv2d(nets, 125, (3, 3), normalizer_fn=None, activation_fn=None,  scope=flower_point[2])
                endpoints[flower_point[2]] = nets
                # tf.contrib.layers.softmax(nets)
    return nets, endpoints


def yoloseparabe(images: tf.Tensor,  depth_multiplier: float, is_training: bool):
    flower_point = ['Conv2d_0_depthwise', 'Conv2d_0_pointwise', 'Conv2d_1_depthwise', 'Conv2d_1_pointwise', 'Final']
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
        nets, endpoints = mobilenet_v1_base(images, depth_multiplier=depth_multiplier)

    # add the new layer
    with tf.variable_scope('Yolo'):
        with slim.arg_scope([slim.batch_norm], is_training=is_training,  center=True, scale=True, decay=0.9997, epsilon=0.001):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d], normalizer_fn=slim.batch_norm, activation_fn=None,):
                # nets=(?, 7, 10, 512)
                nets = slim.separable_conv2d(nets, None, (3, 3), scope=flower_point[0])
                nets = tf.nn.relu6(nets, name=flower_point[0]+'/relu6')
                endpoints[flower_point[0]] = nets
                # nets = (?, 7, 10, 512)
                nets = slim.conv2d(nets, 256, (1, 1), scope=flower_point[1])
                nets = tf.nn.relu6(nets, name=flower_point[1]+'/relu6')
                endpoints[flower_point[1]] = nets
                # nets = (?, 7, 10, 256)
                nets = slim.separable_conv2d(nets, None, (3, 3), scope=flower_point[2])
                nets = tf.nn.relu6(nets, name=flower_point[2]+'/relu6')
                endpoints[flower_point[2]] = nets
                # nets = (?, 7, 10, 256)
                nets = slim.conv2d(nets, 128, (1, 1), scope=flower_point[3])
                nets = tf.nn.relu6(nets, name=flower_point[3]+'/relu6')
                endpoints[flower_point[3]] = nets
                # nets = (?, 7, 10, 128)
                nets = slim.conv2d(nets, 125, (3, 3), normalizer_fn=None, activation_fn=None, scope=flower_point[4])
                endpoints[flower_point[4]] = nets
                # nets = (?, 4, 5, 5)
                # tf.contrib.layers.softmax(nets)
    return nets, endpoints


def yoloconv(images: tf.Tensor, depth_multiplier: float, is_training: bool):
    flower_point = ['Conv2d_0_depthwise', 'Conv2d_0_pointwise', 'Conv2d_1_depthwise', 'Conv2d_1', 'Final']
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
        nets, endpoints = mobilenet_v1_base(images, depth_multiplier=depth_multiplier)

    # add the new layer
    with tf.variable_scope('Yolo'):
        with slim.arg_scope([slim.batch_norm], is_training=is_training,  center=True, scale=True, decay=0.9997, epsilon=0.001):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME', normalizer_fn=slim.batch_norm, activation_fn=None,):
                # (?, 7, 10, 512)
                nets = slim.separable_conv2d(nets, None, (3, 3), scope=flower_point[0])
                nets = tf.nn.relu6(nets, name=flower_point[0]+'/relu6')
                endpoints[flower_point[0]] = nets
                # (?, 7, 10, 512)
                nets = slim.conv2d(nets, 256, (1, 1), scope=flower_point[1])
                nets = tf.nn.relu6(nets, name=flower_point[1]+'/relu6')
                endpoints[flower_point[1]] = nets
                # nets = (?, 7, 10, 256)
                nets = slim.separable_conv2d(nets, None, (3, 3), scope=flower_point[2])
                nets = tf.nn.relu6(nets, name=flower_point[2]+'/relu6')
                endpoints[flower_point[2]] = nets
                # nets = (?, 7, 10, 128)
                nets = slim.conv2d(nets, 128, (3, 3),  scope=flower_point[3])
                nets = tf.nn.relu6(nets, name=flower_point[3]+'/relu6')
                endpoints[flower_point[3]] = nets
                # nets = (?, 7, 10, 128)
                nets = slim.conv2d(nets, 5, (3, 3), normalizer_fn=None, activation_fn=None, scope=flower_point[4])
                endpoints[flower_point[4]] = nets
                # nets = (?, 7, 10, 125)
    return nets, endpoints


def pureconv(images: tf.Tensor, depth_multiplier: float, is_training: bool):
    """ this network input should be 240*320 """
    with tf.variable_scope('Yolo'):
        with tf.variable_scope('convd_1'):
            nets = tf.layers.conv2d(images, 32, (3, 3), padding='same')
            nets = tf.layers.batch_normalization(nets, training=is_training)
            nets = tf.nn.leaky_relu(nets)
            nets = tf.layers.max_pooling2d(nets, pool_size=(2, 2), strides=(2, 2))
            # nets=[120,160,8]
        with tf.variable_scope('convd_2'):
            nets = tf.layers.conv2d(nets, 32, (3, 3), padding='same')
            nets = tf.layers.batch_normalization(nets, training=is_training)
            nets = tf.nn.leaky_relu(nets)
            nets = tf.layers.max_pooling2d(nets, pool_size=(2, 2), strides=(2, 2))
            # nets=[60,80,16]
        with tf.variable_scope('convd_3'):
            nets = tf.layers.conv2d(nets, 64, (3, 3), padding='same')
            nets = tf.layers.batch_normalization(nets, training=is_training)
            nets = tf.nn.leaky_relu(nets)
            nets = tf.layers.max_pooling2d(nets, pool_size=(2, 2), strides=(2, 2))
            # nets=[30,40,32]
        with tf.variable_scope('convd_4'):
            nets = tf.layers.conv2d(nets, 64, (3, 3), padding='same')
            nets = tf.layers.batch_normalization(nets, training=is_training)
            nets = tf.nn.leaky_relu(nets)
            nets = tf.layers.max_pooling2d(nets, pool_size=(2, 2), strides=(2, 2))
            # nets=[15,20,64]
        with tf.variable_scope('convd_5'):
            nets = tf.layers.conv2d(nets, 64, (3, 3), padding='same')
            nets = tf.layers.batch_normalization(nets, training=is_training)
            nets = tf.nn.leaky_relu(nets)
            nets = tf.layers.max_pooling2d(nets, pool_size=(2, 2), strides=(2, 2))
            # nets=[7,10,64]
        with tf.variable_scope('convd_6'):
            nets = tf.layers.conv2d(nets, 32, (3, 3), padding='same')
            nets = tf.layers.batch_normalization(nets, training=is_training)
            nets = tf.nn.leaky_relu(nets)
            # nets=[7,10,64]
        with tf.variable_scope('Final'):
            nets = tf.layers.conv2d(nets, 5, (3, 3), padding='same')
            # nets=[7,10,5]
    endpoints = None
    return nets, endpoints
