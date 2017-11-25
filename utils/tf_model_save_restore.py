# -*- coding:utf-8 -*-

"""
@author: Yan Liu
@file: tf_model_save_restore.py
@time: 2017/11/25 16:06
@desc: Tensorflow模型的保存与恢复加载
"""

import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


def save_model_ckpt(ckpt_file_path):
    x = tf.placeholder(tf.int32, name='x')
    y = tf.placeholder(tf.int32, name='y')
    b = tf.Variable(1, name='b')
    xy = tf.multiply(x, y)
    op = tf.add(xy, b, name='op_to_store')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    path = os.path.dirname(os.path.abspath(ckpt_file_path))
    if os.path.isdir(path) is False:
        os.makedirs(path)

    tf.train.Saver().save(sess, ckpt_file_path + '.ckpt')

    # test
    feed_dict = {x: 2, y: 3}
    print(sess.run(op, feed_dict))


def restore_model_ckpt(ckpt_file_path):
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./ckpt/model.ckpt.meta')  # 加载模型结构
    saver.restore(sess, tf.train.latest_checkpoint('./ckpt'))  # 只需要指定目录就可以恢复所有变量信息

    # 直接获取保存的变量
    print(sess.run('b:0'))

    # 获取placeholder变量
    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')
    # 获取需要进行计算的operator
    op = sess.graph.get_tensor_by_name('op_to_store:0')

    # 加入新的操作
    add_on_op = tf.multiply(op, 2)

    ret = sess.run(add_on_op, {input_x: 5, input_y: 5})
    print(ret)


def save_mode_pb(pb_file_path):
    x = tf.placeholder(tf.int32, name='x')
    y = tf.placeholder(tf.int32, name='y')
    b = tf.Variable(1, name='b')
    xy = tf.multiply(x, y)
    # 这里的输出需要加上name属性
    op = tf.add(xy, b, name='op_to_store')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    path = os.path.dirname(os.path.abspath(pb_file_path))
    if os.path.isdir(path) is False:
        os.makedirs(path)

    # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])
    with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    # test
    feed_dict = {x: 2, y: 3}
    print(sess.run(op, feed_dict))


def restore_mode_pb(pb_file_path):
    sess = tf.Session()
    with gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    print(sess.run('b:0'))

    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')

    op = sess.graph.get_tensor_by_name('op_to_store:0')

    ret = sess.run(op, {input_x: 5, input_y: 5})
    print(ret)


if __name__ == '__main__':
    # save_model_ckpt("./ckpt/model")
    # restore_model_ckpt('./ckpt/model')
    # save_mode_pb('./pb/model.pb')
    restore_mode_pb('./pb/model.pb')