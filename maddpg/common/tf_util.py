# maddpg/common/tf_util_tf2.py (Replacement for tf_util.py)
import tensorflow as tf
import numpy as np
import os

# Most functions from tf_util.py are replaced by native TF2/Keras features:
# - Session management: Gone (Eager execution)
# - Placeholders: Gone (Direct input or tf.keras.Input)
# - U.function: Gone (@tf.function decorator)
# - U.initialize: Gone (Object initialization)
# - scope_vars: Use model.trainable_variables
# - save/load_state: Use tf.train.Checkpoint
# - minimize_and_clip: Use tf.GradientTape and optimizer.apply_gradients (with clipping option)
# - make_session: Use tf.config functions if specific CPU/GPU config is needed

# --- Optional GPU Setup Utility ---
def setup_gpu(device_id="0"):
    """Configures TensorFlow to use a specific GPU and allow memory growth."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            target_gpu = None
            if device_id:
                 # Find the GPU matching the ID (assuming IDs are like "0", "1")
                gpu_indices = [i for i, gpu in enumerate(gpus) if gpu.name.endswith(f':{device_id}')]
                if gpu_indices:
                    target_gpu = gpus[gpu_indices[0]]
                else:
                    print(f"Warning: GPU device ID {device_id} not found. Using default.")
                    target_gpu = gpus[0] # Fallback to first GPU
            else:
                 target_gpu = gpus[0] # Use first GPU if no ID specified

            if target_gpu:
                try:
                    # Set memory growth before setting visible devices
                    tf.config.experimental.set_memory_growth(target_gpu, True)
                    # Set only the target GPU to be visible
                    tf.config.set_visible_devices(target_gpu, 'GPU')
                    logical_gpus = tf.config.list_logical_devices('GPU')
                    print(f"Using GPU: {target_gpu.name}, Logical GPU: {logical_gpus[0].name}")
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(f"Error setting up GPU: {e}")
            else:
                 tf.config.set_visible_devices([], 'GPU') # Explicitly use CPU
                 print("No target GPU found or specified. Using CPU.")

        else:
            print("No GPUs detected. Using CPU.")
    except Exception as e:
        print(f"An error occurred during GPU setup: {e}")

# --- Math utils (can often use tf.* directly) ---
# These are mostly direct replacements or already exist in tf.*

def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    # Use tf.keras.losses.Huber or implement directly
    # return tf.keras.losses.Huber(delta=delta)(tf.zeros_like(x), x) # A bit indirect
    # Direct implementation:
    abs_error = tf.abs(x)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * tf.square(quadratic) + delta * linear

# --- Target Network Update ---
@tf.function
def soft_update_vars(source_vars, target_vars, tau):
    """Performs a soft update (Polyak averaging) of target variables."""
    for source, target in zip(source_vars, target_vars):
        target.assign(tau * source + (1.0 - tau) * target)

# --- Checkpoint Loading Helper (Optional) ---
def load_latest_checkpoint(checkpoint_dir, checkpoint):
    """Loads the latest checkpoint from a directory."""
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        status = checkpoint.restore(latest)
        # status.assert_consumed() # Optional: check restoration status
        print(f"Restored checkpoint from {latest}")
        return status
    else:
        print(f"No checkpoint found in {checkpoint_dir}")
        return None

# Note: The 'function' utility and placeholder helpers are deliberately removed.
# The core logic shifts towards Keras models and @tf.function decorated methods.