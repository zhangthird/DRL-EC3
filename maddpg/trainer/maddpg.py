# maddpg/trainer/maddpg_tf2.py (Replacement for maddpg.py)

import logging
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, LSTM, Input, Flatten, Concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Assuming replay buffer is compatible or updated (uses NumPy primarily)
# Need to import the correct ReplayBuffer class
from maddpg.trainer.prioritized_rb.replay_buffer import ReplayBuffer

# Import TF2 distributions and pdtype factory
# from maddpg.common.distributions import make_pdtype
from maddpg.common.distributions import (
    make_pdtype, # Still needed for agent init
    DiagGaussianPdType,
    CategoricalPdType,
    SoftCategoricalPdType, # If using Gumbel-Softmax
    BernoulliPdType
    # Add any other types you defined, e.g., MultiCategoricalPdType
)

# Import TF2 utilities
from maddpg.common.tf_util import huber_loss, soft_update_vars


# --- Network Definitions using Keras Functional API or Subclassing ---

def build_cnn(input_shape, scope='CNN'):
    """Builds the CNN part of the network."""
    # Using Functional API example
    # Name layers uniquely if necessary, Keras handles scopes well
    img_input = Input(shape=input_shape, name=f'{scope}_input')
    x = Conv2D(16, 3, strides=2, padding='valid', activation='relu', name=f'{scope}_conv1')(img_input)
    x = Conv2D(32, 3, strides=2, padding='valid', activation='relu', name=f'{scope}_conv2')(x)
    x = Conv2D(64, 3, strides=2, padding='valid', activation='relu', name=f'{scope}_conv3')(x)
    x = BatchNormalization(name=f'{scope}_bn')(x) # BN after conv usually
    x = Flatten(name=f'{scope}_flatten')(x)
    # Model can be returned directly if CNN is standalone
    # return Model(inputs=img_input, outputs=x, name=scope)
    # Or just return the output tensor to be used in a larger model
    model = Model(inputs=img_input, outputs=x, name=scope)
    return model


# TODO: RNN needs careful implementation if used
def build_rnn(input_tensor, cell_size, scope='RNN'):
     """Builds an LSTM layer."""
     # Assuming input_tensor has shape [time_steps, batch_size, features]
     # Or [batch_size, time_steps, features] if time_major=False (Keras default)
     # LayerNorm LSTM isn't standard in Keras, may need custom layer or TFA
     # from tensorflow_addons.layers import LayerNormalization # Requires tensorflow-addons
     # lstm_layer = LSTM(cell_size, return_sequences=False, return_state=False, name=f'{scope}_lstm')
     # Use standard LSTM for now
     # Note: Keras LSTM expects batch first by default [batch, timesteps, feature]
     # If input is [timesteps, batch, feature], need to transpose or adjust LSTM layer/input processing
     lstm_out = LSTM(cell_size, return_sequences=False, name=f'{scope}_lstm')(input_tensor) # Gets last output
     # If state needs to be returned:
     # lstm_out, state_h, state_c = LSTM(cell_size, return_sequences=False, return_state=True)(input_tensor)
     return lstm_out #, state_h, state_c # Return state if needed


def build_mlp(input_tensor, num_outputs, scope, num_units=64, activation_fn='relu', output_activation=None, reg_scale=1e-2):
    """Builds the MLP part of the network."""
    reg = l2(reg_scale) if reg_scale > 0 else None
    x = Dense(num_units, activation=activation_fn, kernel_regularizer=reg, bias_regularizer=reg, name=f'{scope}_dense1')(input_tensor)
    x = Dense(num_units, activation=activation_fn, kernel_regularizer=reg, bias_regularizer=reg, name=f'{scope}_dense2')(x)
    x = Dense(num_outputs, activation=output_activation, kernel_regularizer=reg, bias_regularizer=reg, name=f'{scope}_output')(x)
    return x


# --- Actor and Critic Models ---

def build_actor(obs_shape, act_pdtype, num_units, args, name="Actor"):
    """Builds the Actor network model."""
    # 1. 创建 CNN 模型部分
    cnn_model = build_cnn(obs_shape, scope=f'{name}_cnn')
    # 2. 从 CNN 模型获取输入层和输出张量
    cnn_input = cnn_model.input
    cnn_output = cnn_model.output # 这是 CNN 处理后的特征张量

    last_layer = cnn_output # MLP 的输入是 CNN 的输出

    # RNN Part (if used) - Needs careful shape handling
    if args.rnn_length > 0:
        # Input to RNN needs shape [batch, time_steps, features] or [time_steps, batch, features]
        # CNN output is likely [batch, features]. Reshaping or sequence handling needed.
        # Example: assuming cnn_output holds features for one step, needs sequence input
        # This part requires specific logic based on how sequences are fed
        # rnn_input = tf.reshape(cnn_output, [-1, args.rnn_length, cnn_output.shape[-1]]) # If batch contains sequences
        # last_layer = build_rnn(rnn_input, args.rnn_cell_size, scope=f'{name}_rnn')
        raise NotImplementedError("Actor RNN integration needs specific implementation.")

    # MLP Part
    # act_pdtype helps determine output size and activation
    output_size = act_pdtype.param_shape()[0]
    output_activation = None # Default to no activation
    # Check the type of the act_pdtype object directly
    if isinstance(act_pdtype, (CategoricalPdType, BernoulliPdType, SoftCategoricalPdType)):
        # These output logits, so no activation on the final layer
        output_activation = None
    elif isinstance(act_pdtype, DiagGaussianPdType):
        # Outputs mean and logstd, no activation needed here.
        # Use Tanh *bijector* later if actions need bounding.
        output_activation = None
    # Add elif for other custom types if needed, e.g.:
    # elif isinstance(act_pdtype, MultiCategoricalPdType):
    #     output_activation = None
    else:
        # Optional: Log a warning if the type is unexpected
        logging.warning(f"Actor output activation: Unhandled act_pdtype {type(act_pdtype)}, defaulting to None.")


    # Tanh activation was mentioned in comments - apply if needed
    # If using Tanh, often applied *after* the final layer or via TFP's Tanh Bijector
    actor_output = build_mlp(last_layer, output_size, scope=f'{name}_mlp',
                             num_units=num_units, output_activation=output_activation, reg_scale=1e-2)

    # Optional Tanh application (if action space is bounded, e.g., [-1, 1])
    # actor_output = tf.keras.layers.Activation('tanh')(actor_output)
    # --- 创建最终的 Actor 模型 ---
    # 输入是 CNN 模型的输入，输出是 MLP 的输出
    model = Model(inputs=cnn_input, outputs=actor_output, name=name)
    print(f"{name} Model Summary:")
    model.summary()
    # import io
    # summary_stream = io.StringIO()
    # model.summary(print_fn=lambda x: summary_stream.write(x + '\n'))
    # logging.info(f"{name} Model Summary:\n{summary_stream.getvalue()}")
    return model


def build_critic(obs_shapes_n, act_shapes_n, num_units, args, agent_index, local_q_func, name="Critic"):
    """Builds the Critic network model."""
    # Inputs: list of observations and list of actions from all agents
    obs_inputs = [Input(shape=shape, name=f'{name}_obs_{i}') for i, shape in enumerate(obs_shapes_n)]
    act_inputs = [Input(shape=shape, name=f'{name}_act_{i}') for i, shape in enumerate(act_shapes_n)]

    # Process observations (CNN + optional RNN)
    cnn_outputs = []
    processed_obs_inputs_for_concat = [] # Keep original inputs for local Q if needed
    # 处理观测值 (CNN + 可选 RNN)
    for i, obs_input in enumerate(obs_inputs):
        # --- 修改：创建并调用 CNN 模型实例 ---
        # 1. 为当前 agent i 的 obs_shape 构建 CNN 模型
        #    (如果所有 obs_shape 相同，可以在循环外构建一个共享权重的 cnn_model，
        #     但为保险起见，如果形状可能不同，分开构建更安全)
        cnn_model_i = build_cnn(obs_shapes_n[i], scope=f'{name}_cnn_{i}')

        # 2. 使用 agent i 的具体输入张量 obs_input 来调用这个模型实例
        cnn_out = cnn_model_i(obs_input)

        processed_obs = cnn_out
        processed_obs_inputs_for_concat.append(obs_input) # Save original input ref

        # RNN Part (if used)
        if args.rnn_length > 0:
             # Similar RNN integration challenges as in Actor
             # rnn_input = tf.reshape(cnn_out, [-1, args.rnn_length, cnn_out.shape[-1]])
             # processed_obs = build_rnn(rnn_input, args.rnn_cell_size, scope=f'{name}_rnn_{i}')
             raise NotImplementedError("Critic RNN integration needs specific implementation.")
        cnn_outputs.append(processed_obs)

    # Concatenate inputs for the MLP
    if local_q_func:
        # 只连接当前 agent 的 CNN 输出和动作输入
        mlp_input_tensors = [cnn_outputs[agent_index], act_inputs[agent_index]]
        critic_inputs = [obs_inputs[agent_index], act_inputs[agent_index]] # 跟踪原始输入
    else:
        # 连接所有 agents 的 CNN 输出和动作输入
        mlp_input_tensors = cnn_outputs + act_inputs
        critic_inputs = obs_inputs + act_inputs # 跟踪原始输入

    # 使用 Keras Concatenate 层进行连接 (axis=1 表示沿特征维度连接)
    # mlp_input_tensors 是一个包含所有要连接的 Keras symbolic tensors 的列表
    concatenated_inputs = Concatenate(axis=1, name=f'{name}_concat')(mlp_input_tensors)


    # MLP Part
    q_value = build_mlp(concatenated_inputs, 1, scope=f'{name}_mlp', num_units=num_units, output_activation=None, reg_scale=1e-2)

    model = Model(inputs=critic_inputs, outputs=q_value, name=name)
    print(f"{name} Model Summary:")
    model.summary()
    return model


class MADDPGAgentTrainer:
    def __init__(self, name, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.local_q_func = local_q_func
        self.tau = 1e-2 # Polyak averaging coefficient

        # Get observation and action shapes/types for this agent
        self.obs_shape = obs_shape_n[agent_index]
        self.act_space = act_space_n[agent_index]
        self.act_pdtype = make_pdtype(self.act_space) # Distribution type helper

        # Action shape needed for critic input placeholder
        # Use sample shape if available, otherwise infer carefully
        act_shape = None
        try:
            # 优先使用 pdtype 辅助类中定义的 sample_shape 方法
            act_shape = self.act_pdtype.sample_shape()
            if not act_shape: # 处理 sample_shape 返回 [] 或 None 的情况
                act_shape = (1,)
            else:
                act_shape = tuple(act_shape) # 确保是元组
        except AttributeError:
             # 如果辅助类没有 sample_shape 方法，则回退到检查 Gym space
             logging.warning(f"act_pdtype {type(self.act_pdtype)} 没有 sample_shape 方法。回退到 Gym space。")
             if hasattr(self.act_space, 'shape'):
                 act_shape = self.act_space.shape
             elif hasattr(self.act_space, 'n'): # 处理离散空间
                 # --- **修正这里的 isinstance 检查** ---
                 # 直接与导入的辅助类比较，不再调用 make_pdtype
                 if isinstance(self.act_pdtype, SoftCategoricalPdType):
                      # SoftCategorical 输出维度为 n 的向量
                      act_shape = (self.act_space.n,)
                 elif isinstance(self.act_pdtype, CategoricalPdType):
                      # 标准 Categorical 输出标量索引
                      act_shape = (1,)
                 else:
                      # 未知离散空间映射的默认处理
                      act_shape = (1,)
                      logging.warning(f"未知的 Discrete space pdtype ({type(self.act_pdtype)})，假设 action shape 为 (1,)")
             else:
                 raise ValueError(f"无法确定 Agent {agent_index} 的 action shape (space: {self.act_space})")

        if act_shape is None:
             raise ValueError(f"无法确定 Agent {agent_index} 的 action shape")


        # Build Networks
        self.actor_model = build_actor(self.obs_shape, self.act_pdtype, args.num_units, args, name=f"{name}_Actor")
        self.target_actor_model = build_actor(self.obs_shape, self.act_pdtype, args.num_units, args, name=f"{name}_TargetActor")
        self.target_actor_model.set_weights(self.actor_model.get_weights()) # Initialize target nets

        # Critic needs shapes of all obs and actions
        # --- 确定所有 agents 的 action shapes 用于 Critic 输入 ---
        all_act_shapes = []
        for i in range(self.n):
             # 获取 agent i 的 pdtype 辅助对象
             pdtype_i = make_pdtype(act_space_n[i])
             shape_i = None
             try:
                 # 优先使用 sample_shape
                 shape_i = pdtype_i.sample_shape()
                 if not shape_i: shape_i = (1,)
                 else: shape_i = tuple(shape_i)
             except AttributeError:
                  # 回退逻辑
                  logging.warning(f"Agent {i} 的 act_pdtype {type(pdtype_i)} 没有 sample_shape 方法。回退。")
                  if hasattr(act_space_n[i], 'shape'):
                      shape_i = act_space_n[i].shape
                  elif hasattr(act_space_n[i], 'n'):
                       # --- **修正这里的 isinstance 检查** ---
                       # 直接与导入的辅助类比较，不再调用 make_pdtype
                       if isinstance(pdtype_i, SoftCategoricalPdType):
                            shape_i = (act_space_n[i].n,)
                       elif isinstance(pdtype_i, CategoricalPdType):
                            shape_i = (1,)
                       else:
                            shape_i = (1,)
                            logging.warning(f"未知的 Discrete space pdtype ({type(pdtype_i)}) (Agent {i})，假设 action shape 为 (1,)")
                  else:
                       raise ValueError(f"无法确定 Agent {i} (critic input) 的 action shape")

             if shape_i is None:
                  raise ValueError(f"无法确定 Agent {i} (critic input) 的 action shape")
             all_act_shapes.append(shape_i)

        # --- 构建 Critic 网络 ---
        self.critic_model = build_critic(obs_shape_n, all_act_shapes, args.num_units, args, agent_index, local_q_func, name=f"{name}_Critic")
        self.target_critic_model = build_critic(obs_shape_n, all_act_shapes, args.num_units, args, agent_index, local_q_func, name=f"{name}_TargetCritic")
        self.target_critic_model.set_weights(self.critic_model.get_weights()) # Initialize target nets

        # Optimizers
        # Learning rate schedule (optional)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            args.lr,
            decay_steps=10000, # Adjust decay steps
            decay_rate=0.99, # Adjust decay rate
            staircase=True)
        self.actor_optimizer = Adam(learning_rate=lr_schedule, clipnorm=0.5) # Use clipnorm or clipvalue if needed
        self.critic_optimizer = Adam(learning_rate=lr_schedule, clipnorm=0.5)

        # Experience buffer
        # Assuming ReplayBuffer uses NumPy and is compatible
        self.replay_buffer = ReplayBuffer(int(self.args.buffer_size), int(self.args.batch_size), self.args.alpha,
                                          self.args.epsilon)
        self.replay_sample_index = None
        self.beta = self.args.beta # Initial beta for PER

        # Global step tracking (optional, Keras optimizer tracks iterations)
        self.global_train_step = tf.Variable(0, trainable=False, dtype=tf.int64)


    @property
    def filled_size(self):
        return len(self.replay_buffer)

    def is_buffer_ready(self):
         # Heuristic: wait for > N batches before starting updates
         # Original code waited for 100 * batch_size
         return len(self.replay_buffer) > 10 * self.args.batch_size # Reduced threshold


    @tf.function # Decorate action selection for potential speedup
    def action(self, obs_batch):
        """ Get actions for a batch of observations. """
        # Obs_batch shape: [batch_size, obs_dim...]
        actor_output = self.actor_model(obs_batch, training=False) # Get raw network output (e.g., logits, mean/logstd)
        pd = self.act_pdtype.pd_from_flat(actor_output) # Create distribution object

        # Sample action from the distribution
        action = pd.sample()

        # Mode might be preferred for evaluation
        # action = pd.mode()

        # If action space is bounded and model output isn't, apply Tanh Bijector or clipping here
        # Example with Tanh Bijector (assumes pd is Gaussian):
        # base_distribution = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
        # tanh_bijector = tfp.bijectors.Tanh()
        # transformed_distribution = tfp.distributions.TransformedDistribution(
        #     distribution=base_distribution,
        #     bijector=tanh_bijector
        # )
        # action = transformed_distribution.sample()

        return action


    def experience(self, obs, act, rew, new_obs, done, terminal, num_actor_workers):
        # Store transition in the replay buffer.
        # Ensure inputs are NumPy arrays if buffer expects them
        self.replay_buffer.add(obs, act, rew, new_obs, float(done), self.args.N, self.args.gamma, num_actor_workers)


    @tf.function # Compile the update step for performance
    def _perform_update(self, obs_n_batch, act_n_batch, rew_batch, next_obs_n_batch, done_batch, target_act_next_n_batch):
        """Performs the core Critic and Actor update calculations."""
        # Target Critic Q-value calculation
        target_q_input = next_obs_n_batch + target_act_next_n_batch
        if self.local_q_func:
             # Select only local obs/act for target critic input
             target_q_input = [next_obs_n_batch[self.agent_index], target_act_next_n_batch[self.agent_index]]

        target_q_next = self.target_critic_model(target_q_input, training=False)
        target_q = rew_batch + (self.args.gamma ** self.args.N) * (1.0 - done_batch) * tf.squeeze(target_q_next, axis=1)

        # Critic update
        with tf.GradientTape() as tape:
            critic_input = obs_n_batch + act_n_batch
            if self.local_q_func:
                critic_input = [obs_n_batch[self.agent_index], act_n_batch[self.agent_index]]

            current_q = self.critic_model(critic_input, training=True) # Train critic
            current_q = tf.squeeze(current_q, axis=1)

            # Calculate TD error and loss (e.g., Huber loss or MSE)
            td_error = target_q - current_q
            # critic_loss = tf.reduce_mean(huber_loss(td_error)) # Use Huber loss
            critic_loss = tf.reduce_mean(tf.square(td_error)) # Use MSE loss

            # Add regularization losses if any
            if self.critic_model.losses:
                 critic_loss += tf.add_n(self.critic_model.losses)

        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        # Apply gradients (clipping handled by optimizer if configured)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        # Actor update
        with tf.GradientTape() as tape:
            # Get actions from current policy based on sampled observations
            actor_output = self.actor_model(obs_n_batch[self.agent_index], training=True) # Train actor
            pd = self.act_pdtype.pd_from_flat(actor_output)
            new_actions_batch = pd.sample() # Sample actions for policy gradient calculation

            # Create full action list with new actions for this agent
            actor_update_act_n = list(act_n_batch) # Copy original actions
            actor_update_act_n[self.agent_index] = new_actions_batch

            # Calculate Q-value using the *non-target* critic
            critic_input_for_actor = obs_n_batch + actor_update_act_n
            if self.local_q_func:
                 critic_input_for_actor = [obs_n_batch[self.agent_index], actor_update_act_n[self.agent_index]]

            actor_q_values = self.critic_model(critic_input_for_actor, training=False) # IMPORTANT: Don't train critic here

            # Actor loss is negative Q-value (aims to maximize Q)
            actor_loss = -tf.reduce_mean(actor_q_values)

            # Add regularization losses if any
            if self.actor_model.losses:
                 actor_loss += tf.add_n(self.actor_model.losses)

            # Optional: Add entropy bonus if using stochastic policy (e.g., for SAC-like behavior)
            # entropy = tf.reduce_mean(pd.entropy())
            # actor_loss -= entropy_coeff * entropy

        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        # Return losses and TD errors for logging and PER update
        return critic_loss, actor_loss, td_error, target_q, current_q


    def update(self, agents, t, debug_dir):
        # Return if buffer not ready
        if not self.is_buffer_ready():
            return None

        # Update only periodically (e.g., every 10 steps in original code)
        # if t % 10 != 0:
        #     return None
        self.global_train_step.assign_add(1) # Increment step count

        # Update beta for PER
        if self.beta < 1.0:
            self.beta = min(1.0, self.beta * (1.0 + 1e-5)) # Slower increase?

        # Sample from experience replay buffer
        (obs, act, rew, obs_next, done), weights, priorities, self.replay_sample_index = self.replay_buffer.sample(
            self.args.batch_size, self.beta, self.args.num_actor_workers, self.args.rnn_length)

        # Collect samples for all agents based on the sampled indices
        obs_n_batch = []
        act_n_batch = []
        next_obs_n_batch = []
        rew_batch = tf.constant(rew, dtype=tf.float32) # Use reward from this agent's sampled batch
        done_batch = tf.constant(done, dtype=tf.float32) # Use done from this agent's sampled batch

        for i in range(self.n):
            # Use sample_index method from buffer (assuming it exists and is compatible)
            obs_i, act_i, _, next_obs_i, _ = agents[i].replay_buffer.sample_index(
                self.replay_sample_index, self.args.num_actor_workers, self.args.rnn_length)

            # Handle RNN sequence dimension if needed
            # if self.args.rnn_length > 0: ... transpose/reshape ...

            obs_n_batch.append(tf.constant(obs_i, dtype=tf.float32))
            act_n_batch.append(tf.constant(act_i, dtype=tf.float32)) # Ensure correct dtype
            next_obs_n_batch.append(tf.constant(next_obs_i, dtype=tf.float32))


        # Calculate target actions using target actor networks for all agents
        target_act_next_n_batch = []
        for i in range(self.n):
             # Use target actor of agent 'i' with its corresponding next observation batch
             target_actor_output = agents[i].target_actor_model(next_obs_n_batch[i], training=False)
             target_pd = agents[i].act_pdtype.pd_from_flat(target_actor_output)
             target_action_sample = target_pd.sample() # Sample from target policy noise
             target_act_next_n_batch.append(target_action_sample)


        # Perform the update calculation within the @tf.function
        critic_loss, actor_loss, td_error, target_q_val, current_q_val = self._perform_update(
             obs_n_batch, act_n_batch, rew_batch, next_obs_n_batch, done_batch, target_act_next_n_batch
        )

        # Update priorities in the replay buffer
        new_priorities = np.abs(td_error.numpy()) + self.args.epsilon # Add epsilon for non-zero priority
        self.replay_buffer.priority_update(self.replay_sample_index, new_priorities)

        # Soft update target networks
        soft_update_vars(self.actor_model.variables, self.target_actor_model.variables, self.tau)
        soft_update_vars(self.critic_model.variables, self.target_critic_model.variables, self.tau)


        # --- Debug Logging (Optional) ---
        # Be cautious logging inside @tf.function, prefer returning values
        # if t % 100 == 0: # Log less frequently
        #     try:
        #         debug_file = os.path.join(debug_dir, f"debug_info_{self.name}.txt")
        #         with open(debug_file, 'a') as f: # Append mode
        #             print(f"Step: {t}, Agent: {self.name}", file=f)
        #             # Log sample of values (index 0)
        #             print(f"\t Index: {self.replay_sample_index[0]}, Rew: {rew_batch[0].numpy():.2f}, Prio: {new_priorities[0]:.4f}, TargetQ: {target_q_val[0].numpy():.2f}, CurrentQ: {current_q_val[0].numpy():.2f}", file=f)
        #     except Exception as e:
        #          print(f"Error writing debug info for {self.name}: {e}")


        # Return losses for logging in main training loop
        return [critic_loss.numpy(), actor_loss.numpy()] # Return NumPy values