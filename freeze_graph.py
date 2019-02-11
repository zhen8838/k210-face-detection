import tensorflow as tf
from tensorflow.python.framework import graph_util
import sys
import argparse
from models.yolonet import yoloconv


def freeze(input_checkpoint: str, output_graph: str, output_node: str):
    g = tf.get_default_graph()
    ckpt = tf.train.get_checkpoint_state(input_checkpoint)
    inputs = tf.placeholder(dtype=tf.float32, shape=(1, 224, 320, 3), name='Input_image')
    output, _ = yoloconv(inputs, 0.5, is_training=False)
    pred_label = tf.identity(output, name='predict')
    with tf.Session() as sess:
        loader = tf.train.Saver()
        loader.restore(sess, ckpt.model_checkpoint_path)

        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点
        print(inputs)
        print(output)

def main(args):
    freeze(args.ckpt_path, args.pb_path, args.output_node)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('ckpt_path', type=str,
                        help='Path to the ckpt directory.')

    parser.add_argument('pb_path', type=str,
                        help='Path to the .pb file.')

    parser.add_argument('output_node', type=str,
                        help='the graph output node name')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
