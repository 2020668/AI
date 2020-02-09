# import graphviz
# import os
# from data.common_data import DATA_DIR
#
# with open(os.path.join(DATA_DIR, 'tree.dot')) as f:
#     dot_graph = f.read()
# dot = graphviz.Source(dot_graph)
# dot.view()

import tensorflow as tf


def concat():
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[7, 8, 9], [10, 11, 12]]
    c = tf.concat([a, b], axis=1)
    with tf.compat.v1.Session() as sess:
        print(sess.run(c))


# 变量op
def build():
    a = tf.constant([1, 2, 3, 4, 5])
    var = tf.Variable(tf.random_normal([2, 3], mean=0, stddev=1.0))

    # 必须做一步显示的初始化
    init_op = tf.global_variables_initializer()

    # print(a, var)
    # tf.Variable(initial_value=None, name=None, trainable=True)

    with tf.compat.v1.Session() as sess:
        # 必须运行初始化op 否则报错
        sess.run(init_op)

        # 把程序的图结果写入事件文件
        file_writer = tf.compat.v1.summary.FileWriter('./tmp/', graph=sess.graph)
        print(sess.run([a, var]))


if __name__ == '__main__':
    build()

