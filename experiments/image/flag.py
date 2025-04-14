# 替换 tf.app.flags 的导入
from absl import flags as tf_flags
# import tensorflow as tf

# 使用 absl.flags 替代 tf.app.flags
flags = tf_flags
FLAGS = flags.FLAGS

# map info
flags.DEFINE_integer('image_size', 80, 'the size of image')
flags.DEFINE_integer('image_deepth', 2, 'the deepth of image')
flags.DEFINE_integer('wall_value', -1, 'the value of wall')
flags.DEFINE_integer('wall_width', 4, 'the width of wall')
flags.DEFINE_integer('fill_value', -1, 'the value of FillStation')

flags.DEFINE_integer('map_x', 16, 'the length of x-axis')
flags.DEFINE_integer('map_y', 16, 'the length of y-axis')

# 添加以下代码来确保在导入时标志被解析
# 这样当其他模块导入此模块时，标志已经被解析
FLAGS.mark_as_parsed()