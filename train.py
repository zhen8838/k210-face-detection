import tensorflow as tf
from tools.utils import helper
from models.yolonet import yoloconv
from tensorflow.contrib import slim
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import sys
import argparse


def restore_ckpt(sess: tf.Session(), ckptdir: str):
    if 'mobilenet' in ckptdir:
        variables_to_restore = slim.get_model_variables()
        loader = tf.train.Saver([var for var in variables_to_restore if 'MobilenetV1' in var.name])
        loader.restore(sess, ckptdir)
    else:
        ckpt = tf.train.get_checkpoint_state(ckptdir)
        loader = tf.train.Saver()
        loader.restore(sess, ckpt.model_checkpoint_path)


def main(train_list='data/train.list',
         pre_ckpt='mobilenet_v1_0.5_224',
         depth_multiplier=0.5,
         image_size=(224, 320),
         output_size=(7, 10),
         batch_size=32,
         rand_seed=6,
         max_nrof_epochs=3,
         init_learning_rate=0.0005,
         learning_rate_decay_epochs=10,
         learning_rate_decay_factor=1.0,
         pb_name='Training_save.pb',
         log_dir='log'):
    g = tf.get_default_graph()
    tf.set_random_seed(rand_seed)

    """ generate the dataset """
    fddb = helper(train_list, image_size, output_size)
    fddb.set_dataset(batch_size, rand_seed)
    next_img, next_label = fddb.get_iter()
    epoch_step = 2500//batch_size
    """ define the model """
    batch_image = tf.placeholder_with_default(next_img, shape=[None, image_size[0], image_size[1], 3], name='Input_image')
    batch_label = tf.placeholder_with_default(next_label, shape=[None, output_size[0], output_size[1], 5], name='Input_label')
    true_label = tf.identity(batch_label)
    nets, endpoints = yoloconv(batch_image, depth_multiplier, is_training=True)

    """ reshape the model output """
    pred_label = tf.identity(nets, name='predict')
    """ split the label """
    pred_xywh = pred_label[:, :, :, 0:4]
    pred_xywh = tf.nn.sigmoid(pred_xywh)
    pred_confidence = pred_label[:, :, :, 4:5]
    pred_confidence_sigmoid = tf.nn.sigmoid(pred_confidence)

    true_xywh = true_label[:, :, :, 0:4]
    true_confidence = true_label[:, :, :, 4:5]

    obj_mask = true_confidence[:, :, :, 0] > .7

    """ define loss """
    mse_loss = tf.losses.mean_squared_error(labels=tf.boolean_mask(true_xywh, obj_mask), predictions=tf.boolean_mask(pred_xywh, obj_mask))
    obj_loss = 2 * tf.reduce_sum(tf.boolean_mask(tf.nn.sigmoid_cross_entropy_with_logits(labels=true_confidence, logits=pred_confidence), obj_mask))/batch_size
    noobj_loss = tf.reduce_sum(tf.boolean_mask(tf.nn.sigmoid_cross_entropy_with_logits(labels=true_confidence, logits=pred_confidence), tf.logical_not(obj_mask)))/batch_size
    total_loss = obj_loss+noobj_loss+mse_loss

    """ define steps """
    global_steps = tf.train.create_global_step()

    """ define train optimizer """
    current_learning_rate = tf.train.exponential_decay(init_learning_rate, global_steps, epoch_step // learning_rate_decay_epochs,
                                                       learning_rate_decay_factor, staircase=False)

    train_op = tf.train.AdamOptimizer(learning_rate=current_learning_rate).minimize(total_loss, global_step=global_steps)

    """ calc the accuracy """
    pred_obj_num = tf.reduce_sum(tf.cast(pred_confidence_sigmoid > .7, tf.float32))
    confidence_acc = tf.boolean_mask(tf.equal(true_confidence, tf.cast(pred_confidence_sigmoid > .7, tf.float32)), obj_mask)
    confidence_acc = tf.reduce_mean(tf.cast(confidence_acc, tf.float32))
    with tf.Session() as sess:
        saver = tf.train.Saver()
        # init the model and restore the pre-train weight
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # NOTE the accuracy must init local variable
        restore_ckpt(sess, pre_ckpt)
        # define the log and saver
        subdir = os.path.join(log_dir, datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
        writer = tf.summary.FileWriter(subdir, graph=sess.graph)
        # write_arguments_to_file(os.path.join(subdir, 'arguments.txt'))
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('obj_loss', obj_loss)
        tf.summary.scalar('noobj_loss', noobj_loss)
        tf.summary.scalar('mse_loss', mse_loss)
        tf.summary.scalar('pred_obj_num', pred_obj_num)
        tf.summary.scalar('leraning rate', current_learning_rate)
        tf.summary.histogram('p_confidence', tf.sigmoid(pred_confidence))
        merged = tf.summary.merge_all()
        # 使用进度条库
        for i in range(max_nrof_epochs):
            with tqdm(total=epoch_step, bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}{postfix}]', unit=' batch', dynamic_ncols=True) as t:
                for j in range(epoch_step):
                    summary, _, total_l, con_acc,  lr, step_cnt, p_c = sess.run(
                        [merged, train_op, total_loss, confidence_acc, current_learning_rate, global_steps, pred_confidence_sigmoid])
                    writer.add_summary(summary, step_cnt)
                    t.set_postfix(loss='{:<5.3f}'.format(total_l), con_acc='{:<4.2f}%'.format(con_acc*100), lr='{:7f}'.format(lr))
                    t.update()
        saver.save(sess, save_path=os.path.join(subdir, 'model.ckpt'), global_step=global_steps)
        """ save as pb """
        constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['predict'])
        with tf.gfile.GFile(pb_name, 'wb') as f:
            f.write(constant_graph.SerializeToString())
        print('save over')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_list',                 type=str,   help='trian file lists',           default='data/train.list')
    parser.add_argument('--pre_ckpt',                   type=str,   help='pre-train ckpt dir',         default='mobilenet_v1_0.5_224/mobilenet_v1_0.5_224.ckpt')
    parser.add_argument('--depth_multiplier',           type=float, help='mobilenet depth_multiplier', default=0.5)
    parser.add_argument('--image_size',                 type=tuple, help='net work input image size',  default=(224, 320))
    parser.add_argument('--output_size',                type=tuple, help='net work output image size', default=(7, 10))
    parser.add_argument('--batch_size',                 type=int,   help='batch size',                 default=32)
    parser.add_argument('--rand_seed',                  type=int,   help='random seed',                default=6)
    parser.add_argument('--max_nrof_epochs',            type=int,   help='epoch num',                  default=3)
    parser.add_argument('--init_learning_rate',         type=float, help='init learing rate',          default=0.0005)
    parser.add_argument('--learning_rate_decay_epochs', type=int,   help='learning rate decay epochs', default=10)
    parser.add_argument('--learning_rate_decay_factor', type=int,   help='learning rate decay factor', default=1.0)
    parser.add_argument('--pb_name',                    type=str,   help='pb name',                    default='Training_save.pb')
    parser.add_argument('--log_dir',                    type=str,   help='log dir',                    default='log')

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args.train_list,
         args.pre_ckpt,
         args.depth_multiplier,
         args.image_size,
         args.output_size,
         args.batch_size,
         args.rand_seed,
         args.max_nrof_epochs,
         args.init_learning_rate,
         args.learning_rate_decay_epochs,
         args.learning_rate_decay_factor,
         args.pb_name,
         args.log_dir)
