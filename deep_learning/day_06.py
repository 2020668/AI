# -*- coding: utf-8 -*-
"""

=================================
Author: keen
Created on: 2020/2/10

E-mail:keen2020@outlook.com

=================================


"""


import tensorflow as tf
import os


def same_read():
    # 模拟同步先处理数据 然后才能取数据训练

    # 1. 定义队列
    Q = tf.FIFOQueue(3, tf.float32)

    # 放入数据 注意格式 [0.1, 0.2, 0.3] 会被当做张量
    enq_many = Q.enqueue_many([[0.1, 0.2, 0.3], ])

    # 2. 定义读取数据的过程 取数据 + 1 入队列
    out_q = Q.dequeue()
    data = out_q + 1
    en_q = Q.enqueue(data)

    with tf.compat.v1.Session() as sess:
        # 初始化队列
        sess.run(enq_many)

        # 处理数据
        for i in range(100):
            sess.run(en_q)

        # 训练数据
        for i in range(Q.size().eval()):
            print(sess.run(Q.dequeue()))


# 模拟异步子线程存入样本 主线程读取样本
def diff_read():
    # 1. 定义一个队列 1000
    Q = tf.FIFOQueue(1000, tf.float32)

    # 2. 定义子线程 循环值 + 1 放入队列中
    var = tf.Variable(0.0)
    # 实现一个自增 tf.assign_add
    data = tf.assign_add(var, tf.constant(1.0))
    en_q = Q.enqueue(data)

    # 3. 定义队列管理器op 指定多少个子线程以及分配的任务
    qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 2)

    # 初始化变量的op
    init_op = tf.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)

        # 开启线程管理器
        coord = tf.train.Coordinator()

        # 开启子线程
        threads = qr.create_threads(sess, coord=coord, start=True)

        # 主线程不断读取数据训练
        for i in range(300):
            print(sess.run(Q.dequeue()))

        # 回收线程
        coord.request_stop()
        coord.join(threads)


def read_csv(file_list):
    """
    :param file_list: 文件路径和名字的列表
    :return: 读取的内容
    """
    # 1. 构造文件的队列
    file_queue = tf.train.string_input_producer(file_list)

    # 2. 构造csv阅读器读取队列数据（按一行）
    reader = tf.TextLineReader()
    key, value = reader.read(file_queue)
    print(value)

    # 对每行内容进行解码 record_defaults 指定每一个样本每一列的类型 指定默认值
    # 实际只有两列 填入1表示int类型，默认值1
    records = [['None'], ['None']]
    # 两列使用两个变量接收
    example, label = tf.decode_csv(value, record_defaults=records)

    # 想要读取多个数据 就需要批处理
    example_batch, label_batch = tf.train.batch([example, label], batch_size=20, num_threads=1, capacity=9)
    print(example_batch, label_batch)

    return example_batch, label_batch


if __name__ == '__main__':
    # same_read()
    # diff_read()
    # 1. 找到文件 放入列表 路径+名字 列表当中
    file_name = os.listdir('./tmp/csv_data/')
    file_list = [os.path.join('./tmp/csv_data/', file) for file in file_name]
    # print(file_name)
    example_batch, label_batch = read_csv(file_list)
    # 开启会话运行结果
    with tf.compat.v1.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()
        # 开启读取文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 打印读取的内容
        print(sess.run([example_batch, label_batch]))

        # 回收子线程
        coord.request_stop()
        coord.join(threads)
