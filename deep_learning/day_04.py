import os
import tensorflow as tf


# 屏蔽警告日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def demo():
    # 创建一张图 包含了一组op和tensor 上下文环境
    # 只要使用tensorflow定义的函数都是op
    # 张量tensor 指代数据
    g = tf.Graph()
    print(g)
    with g.as_default():
        c = tf.constant(11.0)
        print(c.graph)

    # 实现一个加法运算
    a = tf.constant(5.0)
    b = tf.constant(6.0)
    sum1 = tf.add(a, b)
    # print(a, b)

    # 默认这张图 相当于给程序分配内存
    graph = tf.compat.v1.get_default_graph()
    print(graph)

    # 一次只能运行一个图结构 可在会话中指定图运行
    # 只要有会话的上下文环境，就可以使用eval取出值了

    # 训练模型 实时提供数据进行训练 指定几行几列 也可定为None 动态适应
    plt = tf.placeholder(tf.float32, [None, 3])
    plt.set_shape([3, 3])
    print(plt)

    with tf.compat.v1.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        print(sess.run(plt, feed_dict={plt: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}))
        print(sess.run(sum1))
        print(sess.run([a, b, sum1]))
        print(a.graph)
        print(sum1.graph)
        print(sess.graph)


def my_regression():
    """
    自实现一个线性回归
    :return: None
    """
    # 1. 准备数据 x 特征值[100, 10] y 目标值[100]
    with tf.variable_scope('data'):
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name='x_data')
        # 矩阵相乘必须是二维的
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope('model'):
        # 2. 建立线性回归模型 1个特征 1个权重 1个偏置 y = x w + b
        # 随机给一个权重和偏置，让它去计算损失，然后优化
        # 用变量定义才能优化 trainable参数 决定这个变量是否跟着梯度下降一起优化
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name='w')
        bias = tf.Variable(0.0, name='b')
        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope('loss'):
        # 3. 建立损失函数 均方误差
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    with tf.variable_scope('optimizer'):
        # 4. 梯度下降优化损失 learning_rate = 0 ~ 1
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 定义一个初始化变量op
    init_op = tf.global_variables_initializer()

    # 通过会话运行程序
    with tf.compat.v1.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 打印随机最先初始化的权重和偏置
        print('随机初始化的权重为:{}, 偏置为:{}'.format(weight.eval(), bias.eval()))

        # 建立事件文件
        file_writer = tf.compat.v1.summary.FileWriter('./tmp', graph=sess.graph)

        # 循环训练 运行优化op
        for i in range(1000):
            sess.run(train_op)
            print('第{}次运行,参数的权重为:{}, 偏置为:{}'.format(i, weight.eval(), bias.eval()))

    return None


if __name__ == '__main__':
    my_regression()
