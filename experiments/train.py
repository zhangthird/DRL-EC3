# Modified experiments/train.py (TensorFlow 2.x)

import argparse
import numpy as np
import os
import tensorflow as tf
import time
import logging # Use standard logging

# Assuming experiments.env0 and maddpg.common.summary have been updated or replaced
# from experiments.env0 import log0 as Log # Needs update/replacement
from experiments.env0.data_collection0 import Env # Assuming Env uses NumPy/standard Python
from maddpg.trainer.maddpg import MADDPGAgentTrainer # Import the TF2 version
from maddpg.common.tf_util import setup_gpu # Optional: GPU setup utility

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set CUDA device (consider doing this outside the script or using TF functions if needed)
# setup_gpu("0") # Example using a utility function

# Hyperparameters (Keep as is, argparse handles them)
ARGUMENTS = [
    # Environment
    ["--scenario", str, "simple_adversary", "name of the scenario script"],
    ["--max-episode-len", int, 500, "maximum episode length"],
    ["--num-episodes", int, 50000, "number of episodes"], # Increased for typical RL
    ["--num-adversaries", int, 0, "number of adversaries(enemy)"],
    ["--good-policy", str, "maddpg", "policy for good agents"],
    ["--adv-policy", str, "maddpg", "policy of adversaries"],

    # Core training parameters
    ["--lr", float, 5e-4, "learning rate for Adam optimizer"],
    # ["--decay_rate", float, 0.99995, "learning rate exponential decay"], # Use Keras schedules instead
    ["--gamma", float, 0.95, "discount factor"],
    ["--batch-size", int, 128, "number of samples per optimization step"], # Adjusted batch size
    ["--num-units", int, 256, "number of units in the mlp hidden layers"], # Adjusted num units

    # Priority Replay Buffer (weights not used)
    ["--alpha", float, 0.5, "priority parameter"],
    ["--beta", float, 0.4, "IS parameter"],
    ["--epsilon", float, 0.01, "a small positive constant for priorities"], # Adjusted epsilon
    ["--buffer_size", int, 100000, "buffer size for each agent"], # Adjusted buffer size

    # N-steps
    ["--N", int, 5, "steps of N-step"],

    # TODO: Experiments (These might require more specific TF2 implementation details)
    # Ape-X (If used, requires significant changes for distributed actors)
    ["--num_actor_workers", int, 1, # Default to 1 (no ApeX)
     "number of environments one agent can deal with. if >1, implement parallel envs"],
    ["--debug_dir", str, "/debug_list/",
     "save index,reward(n-step),priority, value,wi per every sample from experience"],

    # RNN
    ["--rnn-length", int, 0, # Renamed for clarity
     "time_step in rnn. if ==0, not use rnn; else, use rnn."],
    ["--rnn-cell-size", int, 64, "LSTM-cell output's size"],

    # Checkpointing
    ["--exp-name", str, "maddpg_tf2_run", "name of the experiment for logging/saving"],
    ["--save-dir", str, "./results/policy/", "directory in which training state and model should be saved"],
    ["--log-dir", str, "./results/logs/", "directory for TensorBoard logs"],
    ["--save-rate", int, 1000, "save model once every time this many episodes are completed"], # Save less frequently
    ["--load-dir", str, None, # Default to None, set path to load a checkpoint
     "directory/file from which training state and model are loaded (e.g., ./results/policy/ckpt-10)"],

    # Evaluation / Display
    ["--display", bool, False, "render the environment"], # Use bool type
    # Benchmark arguments removed for simplicity, can be added back if needed

    # Training
    ["--random_seed", int, 0, "random seed"]
]


# 参数调节器
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    for arg in ARGUMENTS:
        # Handle boolean arguments specifically
        if arg[1] == bool:
             parser.add_argument(arg[0], type=lambda x: (str(x).lower() == 'true'), default=arg[2], help=arg[3])
        else:
            parser.add_argument(arg[0], type=arg[1], default=arg[2], help=arg[3])
    # Remove old action-based flags if covered by type=bool now
    # Example: parser.add_argument("--display", action="store_true", default=False) -> use type=bool
    return parser.parse_args()

def get_trainers(env, num_adversaries, obs_shape_n, act_space_n, arglist):
    trainers = []
    # Trainer = MADDPGAgentTrainer_TF2 # Make sure to use the TF2 version

    # Adversary agents
    for i in range(num_adversaries):
        # Note: local_q_func might not be directly supported or needed in the same way
        trainers.append(MADDPGAgentTrainer(
            "agent_%d" % i, obs_shape_n, act_space_n, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg'))) # Keep logic if needed

    # Good agents
    for i in range(num_adversaries, env.n):
        trainers.append(MADDPGAgentTrainer(
            "agent_%d" % i, obs_shape_n, act_space_n, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg'))) # Keep logic if needed

    return trainers

# experiments/train.py
def setup_logging_and_saving(arglist):
    """Sets up directories and TensorBoard writer."""
    exp_base_dir = arglist.log_dir
    save_base_dir = arglist.save_dir
    # Use a default name if exp_name is None or empty
    exp_name = arglist.exp_name if arglist.exp_name else time.strftime("run_%Y%m%d_%H%M%S")

    exp_dir = os.path.join(exp_base_dir, exp_name)
    save_dir = os.path.join(save_base_dir, exp_name)
    debug_dir_name = arglist.debug_dir.strip('/\\') # Get the base name
    debug_dir = os.path.join(exp_dir, debug_dir_name) # Put debug inside exp_dir

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    writer = tf.summary.create_file_writer(exp_dir) # Create writer WITH exp_dir

    # Save arguments
    try:
        with open(os.path.join(exp_dir, 'args.txt'), 'w') as f:
            import json
            json.dump(arglist.__dict__, f, indent=2)
    except Exception as e:
        logging.error(f"Could not save arguments: {e}")

    return writer, save_dir, debug_dir, exp_dir


def train(arglist):
    # Set random seeds
    np.random.seed(arglist.random_seed)
    tf.random.set_seed(arglist.random_seed)

    # Create environment (only one for now, parallelism needs explicit implementation)
    # TODO: Implement handling for num_actor_workers > 1 if Ape-X-like parallelism is desired
    if arglist.num_actor_workers > 1:
        logging.warning("num_actor_workers > 1 not fully implemented for TF2 eager. Running with 1 worker.")
        arglist.num_actor_workers = 1

    # Setup Logging and Saving
    writer, save_dir, debug_dir, experiment_log_dir = setup_logging_and_saving(arglist)
    # Get the actual log directory path from the writer

    logging.info("Starting training with arguments:")
    for k, v in sorted(arglist.__dict__.items()):
        logging.info(f"\t{k}: {v}")
    logging.info(f"TensorBoard log directory: {experiment_log_dir}")
    logging.info(f"Checkpoint save directory: {save_dir}")
    logging.info(f"Debug directory: {debug_dir}")

    # Create environment instance, passing the log directory path
    logging.info(f"Initializing environment with log_dir: {experiment_log_dir}")
    try:
        env = Env(log_dir=experiment_log_dir)
    except Exception as e:
         logging.error(f"Failed to initialize environment: {e}")
         logging.error("Ensure Env class in data_collection0.py accepts 'log_dir' argument in __init__.")
         return # Exit if env creation fails

    # Create agent trainers
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    # Need action spaces for trainer initialization
    act_space_n = [env.action_space[i] for i in range(env.n)]
    num_adversaries = min(env.n, arglist.num_adversaries)

    trainers = get_trainers(env, num_adversaries, obs_shape_n, act_space_n, arglist)

    # --- Checkpointing Setup ---
    # Create checkpoints for agents' networks and optimizers
    checkpoints = {f'agent_{i}': tf.train.Checkpoint(actor=trainers[i].actor_model,
                                                    critic=trainers[i].critic_model,
                                                    target_actor=trainers[i].target_actor_model,
                                                    target_critic=trainers[i].target_critic_model,
                                                    actor_optimizer=trainers[i].actor_optimizer,
                                                    critic_optimizer=trainers[i].critic_optimizer)
                   for i in range(env.n)}
    # Add global step or other variables if needed: checkpoints['global_step'] = trainers[0].global_train_step
    manager = tf.train.CheckpointManager(tf.train.Checkpoint(**checkpoints), save_dir, max_to_keep=5)

    # Restore from checkpoint if specified
    if arglist.load_dir:
        status = tf.train.Checkpoint(**checkpoints).restore(arglist.load_dir)
        # status.assert_consumed() # Optional: check if all variables were restored
        logging.info(f"Restored models from {arglist.load_dir}")
        # Extract step number from path if possible, otherwise start from 0 or agent's internal step
        try:
            # A simple way assuming ckpt names like 'ckpt-1000'
            initial_episode = int(arglist.load_dir.split('-')[-1]) * arglist.save_rate # Approximate
            # Or better: load global step from checkpoint if saved
        except:
            initial_episode = 0
            logging.warning("Could not determine episode number from load_dir, starting count from 0.")

    else:
        initial_episode = 0
        logging.info("Starting training from scratch.")


    # --- Training Loop ---
    episode_rewards = [0.0]  # total reward for the current episode
    agent_rewards = [[0.0] for _ in range(env.n)] # individual agent reward
    # final_ep_rewards = [] # No longer needed here, use TensorBoard
    # final_ep_ag_rewards = [] # No longer needed here, use TensorBoard
    obs_n = env.reset() # Get initial observations
    episode_step = 0
    global_total_step = 0 # Total environment steps across all episodes

    # RNN state management (if using RNN)
    # Requires careful handling based on how RNNs are implemented in MADDPGAgentTrainer_TF2
    rnn_states = None # Placeholder, initialize if needed

    t_start = time.time()
    m_time = t_start # Time per episode

    logging.info(f'Starting iterations from episode {initial_episode}...')

    for episode_count in range(initial_episode, arglist.num_episodes):
        episode_step = 0
        obs_n = env.reset()
        current_episode_reward = 0.0
        current_agent_rewards = [0.0] * env.n
        # Reset RNN states at the start of each episode if needed
        # rnn_states = ...

        terminal = False # Episode termination flag
        done_n = [False] * env.n # Individual agent done flags (if applicable)

        while not terminal:
            global_total_step += 1
            episode_step += 1

            # Get actions from agents
            # Handle RNN input/state properly if rnn_length > 0
            if arglist.rnn_length > 0:
                # This part needs careful implementation based on how RNN state is managed
                # action_n, rnn_states = zip(*[agent.action_rnn(obs, state) for agent, obs, state in zip(trainers, obs_n, rnn_states)])
                raise NotImplementedError("RNN logic needs careful TF2 implementation in agent and training loop.")
            else:
                # Get actions for all agents based on current observations
                action_n = [agent.action(np.array(obs)[np.newaxis]) for agent, obs in zip(trainers, obs_n)]
                # Squeeze actions if necessary, depending on model output shape
                action_n = [act[0] for act in action_n] # Assuming action returns shape (1, act_dim)


            # Environment step
            # Assumes env.step returns: new_obs_n, rew_n, done_n, info_n
            # Removed indicator logic for simplicity, add back if needed
            step_return_values = env.step(action_n)
            if len(step_return_values) == 5:
                # 假设第 3 个返回值现在是 self.dn 列表 (来自第1步的修改)
                # 假设第 5 个返回值是 indicator (如果不用，可以改成 _ )
                new_obs_n, rew_n, done_n, info_n, indicator_val = step_return_values # 或者 _, 如果 indicator 不用

                # 确认 done_n 现在是列表/数组
                if not isinstance(done_n, (list, np.ndarray)):
                    logging.error(f"Error: done_n received from env.step is not a list/array! Got: {done_n}")
                    # 可能需要进一步处理错误或停止
            else:
                # 如果返回值数量仍然不匹配，抛出错误
                raise ValueError(f"env.step 返回了 {len(step_return_values)} 个值, 但预期是 5 个。请检查 data_collection0.py 中的 Env.step 返回语句。")

            # --- 后续代码使用 done_n 时，它现在应该是正确的列表/数组 ---
            is_done = all(done_n) if isinstance(done_n, (list, np.ndarray)) else done_n # 检查所有 agent 是否都完成
            terminal = (episode_step >= arglist.max_episode_len) or is_done # 终止条件
            # Update observations
            obs_n = new_obs_n

            # Update rewards
            current_episode_reward += sum(rew_n)
            for i in range(env.n):
                current_agent_rewards[i] += rew_n[i]

            # Update agents (perform optimization step)
            losses = {} # Dictionary to store losses per agent
            # Only update if buffer has enough samples and based on frequency
            # Check buffer size inside agent.update or here
            can_update = all(t.is_buffer_ready() for t in trainers)
            if can_update and global_total_step % 10 == 0: # Example update frequency
                 for i, agent in enumerate(trainers):
                    agent_losses = agent.update(trainers, global_total_step, debug_dir) # Pass other trainers if needed
                    if agent_losses: # Update returns losses if update happened
                        losses[f'agent_{i}_q_loss'] = agent_losses[0]
                        losses[f'agent_{i}_p_loss'] = agent_losses[1]
                        # Add other tracked values if needed

            # Render environment if display flag is set
            if arglist.display:
                env.render()
                time.sleep(0.05) # Slow down rendering

            # End of step

        # --- End of Episode ---
        episode_duration = time.time() - m_time
        m_time = time.time() # Reset episode timer

        # Log episode results to console
        logging.info(f"Episode: {episode_count+1}/{arglist.num_episodes}, "
                     f"Steps: {episode_step}, Total Steps: {global_total_step}, "
                     f"Reward: {current_episode_reward:.2f}, Duration: {episode_duration:.2f}s")
        agent_reward_str = ", ".join([f"{r:.2f}" for r in current_agent_rewards])
        logging.info(f"\tAgent Rewards: [{agent_reward_str}]")
        # Log custom env info if available (e.g., collisions, collection)
        # Example: logging.info(f"\tCollisions: {info_n.get('collisions', 'N/A')}")


        # Log to TensorBoard
        with writer.as_default():
            tf.summary.scalar("reward/episode_total_reward", current_episode_reward, step=episode_count)
            for i in range(env.n):
                tf.summary.scalar(f"reward/agent_{i}_reward", current_agent_rewards[i], step=episode_count)
                if f'agent_{i}_q_loss' in losses:
                    tf.summary.scalar(f"loss/agent_{i}_q_loss", losses[f'agent_{i}_q_loss'], step=global_total_step)
                    tf.summary.scalar(f"loss/agent_{i}_p_loss", losses[f'agent_{i}_p_loss'], step=global_total_step)
            # Log buffer size, beta, etc.
            tf.summary.scalar("buffer/agent_0_buffer_size", trainers[0].filled_size, step=global_total_step)
            tf.summary.scalar("params/beta", trainers[0].beta, step=global_total_step) # Assuming beta is tracked in agent
            # tf.summary.scalar("params/learning_rate", trainers[0].actor_optimizer.lr(trainers[0].actor_optimizer.iterations).numpy(), step=global_total_step) # Log current LR

            if trainers: # 确保 trainers 列表非空
                tf.summary.scalar("buffer/agent_0_buffer_size", trainers[0].filled_size, step=global_total_step)
                tf.summary.scalar("params/beta", trainers[0].beta, step=global_total_step)

                # --- 再次修正学习率记录逻辑 ---
                try:
                    # 直接访问优化器的 learning_rate 属性/变量
                    # 根据警告信息，这应该会返回当前的 EagerTensor
                    current_lr = trainers[0].actor_optimizer.learning_rate

                    # 确保它是一个 Tensor (EagerTensor 也是 Tensor)
                    if tf.is_tensor(current_lr):
                        # 直接记录这个 Tensor
                        tf.summary.scalar("params/learning_rate", current_lr, step=global_total_step)
                    else:
                        # 如果意外地不是 Tensor，尝试转换（可能性较低）
                        logging.warning(f"学习率不是 Tensor 类型，尝试从 {type(current_lr)} 转换")
                        tf.summary.scalar("params/learning_rate", tf.cast(current_lr, tf.float32), step=global_total_step)

                except AttributeError:
                     # 如果 .learning_rate 属性不存在（对于 Keras Adam 不太可能）
                     try:
                         # 尝试访问内部超参数（API 可能不稳定）
                         current_lr_val = trainers[0].actor_optimizer._get_hyper('learning_rate')
                         tf.summary.scalar("params/learning_rate", current_lr_val, step=global_total_step)
                     except Exception as e_hyper:
                          logging.error(f"无法使用 .learning_rate 或 _get_hyper 记录学习率: {e_hyper}")
                except Exception as e:
                    # 捕获其他可能的错误
                    logging.error(f"记录学习率时出错: {e}")
            
            # Log custom env metrics from info_n if needed
            # Example: tf.summary.scalar("env/collisions", info_n.get('collisions', 0), step=episode_count)
            # Example: tf.summary.scalar("env/data_collected", info_n.get('collection', 0), step=episode_count)

        writer.flush()

        # Save model periodically
        if (episode_count + 1) % arglist.save_rate == 0:
            save_path = manager.save(checkpoint_number=episode_count + 1)
            logging.info(f"Saved checkpoint for episode {episode_count + 1}: {save_path}")

    # --- End of Training ---
    logging.info("Training finished.")
    # Final save
    save_path = manager.save(checkpoint_number=arglist.num_episodes)
    logging.info(f"Saved final checkpoint: {save_path}")
    writer.close()
    env.close() # Close environment if necessary

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)