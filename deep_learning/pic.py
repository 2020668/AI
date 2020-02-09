import tensorflow as tf

print(tf.__version__)
# 输出'2.0.0-alpha0'
print(tf.test.is_gpu_available())
# 会输出True,则证明安装成功
