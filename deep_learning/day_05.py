import os
import tensorflow as tf
from tensorflow_core.examples.tutorials.mnist import input_data


# 屏蔽警告日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mst():
    mnist = input_data.read_data_sets("./ST_data/", one_hot=True)

    # 获取图片的特征值
    train_images_data = mnist.train.images      # []可查看某个图的数据

    train_labels_data = mnist.train.labels
    # print(train_images_data)
    # print(train_labels_data)

    # 批次获取50张图片的数据  特征值 + 目标值
    data = mnist.train.next_batch(50)


# 单层 全链接层 实现手写数字识别
# 定义数据占位符 特征值[None, 784]    目标值[None, 10]
# 建立模型 随机初始化权重和偏置  w=[784, 10]   b=[10]

def full_connected():

    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

    # 1. 建立数据占位符
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.int32, [None, 10])

    # 2. 建立一个全链接层的神经网络
    with tf.variable_scope("fc_model"):
        # 随机初始化权重和偏置
        weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name="w")

        bias = tf.Variable(tf.constant(0.0, shape=[10]))

        # 预测None个样本的输出结果 matrix [None, 784] * [784, 10] + [10] = [None, 10}
        y_predict = tf.matmul(x, weight) + bias

    # 3. 求出所以样本的损失值 然后求平均值
    with tf.variable_scope("soft_cross"):
        # 求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4. 梯度下降求出损失
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 5. 计算准确率
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))

        # equal_list  None个样本  [1, 0, 0, 1, 1, ...]
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 收集变量 单个数字值收集
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", accuracy)

    # 高维度变量收集
    tf.summary.histogram("weights", weight)
    tf.summary.histogram("biases", bias)

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 定义一个合并变量的op
    merged = tf.summary.merge_all()

    # 开启会话 去训练
    with tf.compat.v1.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 建立events文件 然后写入
        file_writer = tf.summary.FileWriter("./tmp/", graph=sess.graph)

        # 迭代步数去训练 更新参数预测
        for i in range(2000):
            # 取出真实存在的特征值和目标值
            mnist_x, mnist_y = mnist.train.next_batch(50)

            # 运行train_op训练
            sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

            # 写入每步训练的值
            summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})

            file_writer.add_summary(summary, i)

            print("训练第{}步, 准确率为:{}".format(i, sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))

# 通过tensorboard --logdir='./tmp1/'


if __name__ == '__main__':
    full_connected()

