from __future__ import print_function
import tensorflow as tf

# 检测TF版本
if hasattr(tf, 'function'):
    # TF 2.x
    tf1 = tf.compat.v1
else:
    # TF 1.x
    tf1 = tf

class Summary:
    def __init__(self, session, dir_summary):
        self.__sess = session
        self.__vars = {}
        self.__ops = None
        self.__dir = dir_summary
        self.__writer = tf1.summary.FileWriter(dir_summary, session.graph)

    def add_variable(self, var, name="name"):
        tf1.summary.scalar(name, var)
        assert name not in self.__vars, "Already has " + name
        self.__vars[name] = var

    def build(self):
        self.__ops = tf1.summary.merge_all()

    def run(self, feed_dict, step):
        feed_dict_final = {}
        for key, val in feed_dict.items():
            feed_dict_final[self.__vars[key]] = val
        str_summary = self.__sess.run(self.__ops, feed_dict_final)
        self.__writer.add_summary(str_summary, step)
        self.__writer.flush()



