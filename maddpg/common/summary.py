# maddpg/common/summary_tf2.py (Replacement for summary.py)
# This file might not be needed anymore if TF2 summary API is used directly in train.py

# If you still want a wrapper class (less common in TF2):
import tensorflow as tf
import os
import time

class SummaryWriter:
    def __init__(self, log_dir, exp_name):
        """
        Initializes a TensorFlow 2 summary writer.

        Args:
            log_dir: Base directory for logs.
            exp_name: Specific experiment name for the sub-directory.
        """
        log_path = os.path.join(log_dir, exp_name if exp_name else time.strftime("%Y%m%d-%H%M%S"))
        self.writer = tf.summary.create_file_writer(log_path)
        print(f"Initialized SummaryWriter at: {log_path}")

    def write_scalar(self, name, value, step):
        """Writes a scalar value to TensorBoard."""
        with self.writer.as_default():
            tf.summary.scalar(name, value, step=step)

    def flush(self):
        """Flushes the writer buffer."""
        self.writer.flush()

    def close(self):
        """Closes the writer."""
        self.writer.close()

    def get_logdir(self):
        """Returns the log directory path."""
        return self.writer.get_logdir()

# Usage in train.py would be:
# from maddpg.common.summary_tf2 import SummaryWriter
# writer = SummaryWriter(arglist.log_dir, arglist.exp_name)
# ...
# writer.write_scalar("reward/episode_total_reward", current_episode_reward, step=episode_count)
# ...
# writer.flush()
# ...
# writer.close()

# However, the direct usage shown in the modified train.py is often preferred:
# writer = tf.summary.create_file_writer(exp_dir)
# with writer.as_default():
#     tf.summary.scalar(...)
# writer.flush()
# writer.close()