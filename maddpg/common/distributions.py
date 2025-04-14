# maddpg/common/distributions_tf2.py (Replacement for distributions.py)

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from gym import spaces # Keep gym dependency if needed for make_pdtype

# Consider using TensorFlow Probability (TFP) for distributions.
# TFP provides robust implementations of distributions and their methods.

# Base classes Pd and PdType might become redundant if using TFP directly.
# If you need the abstraction, adapt them to wrap TFP distributions.

# --- Distribution Types using TFP ---

class DiagGaussianPdType:
    def __init__(self, size):
        self.size = size

    def pd_from_flat(self, flat):
        """Creates a TFP MultivariateNormalDiag distribution."""
        mean, logstd = tf.split(flat, 2, axis=-1)
        # Ensure logstd is capped for stability if needed
        # logstd = tf.clip_by_value(logstd, -20, 2)
        std = tf.exp(logstd)
        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)

    def sample_placeholder(self, prepend_shape, name=None): # Not needed in TF2
        pass
    def param_placeholder(self, prepend_shape, name=None): # Not needed in TF2
        pass
    def param_shape(self): # Info only
        return [2 * self.size]
    def sample_shape(self): # Info only
        return [self.size]
    def sample_dtype(self): # Info only
        return tf.float32

class CategoricalPdType:
    def __init__(self, ncat):
        self.ncat = ncat

    def pd_from_flat(self, flat):
        """Creates a TFP Categorical distribution from logits."""
        # 'flat' are the logits
        return tfp.distributions.Categorical(logits=flat)

    def param_shape(self): # Info only
        return [self.ncat]
    def sample_shape(self): # Info only
        return []
    def sample_dtype(self): # Info only
        return tf.int32

class SoftCategoricalPdType: # Represents Gumbel-Softmax / Concrete
     def __init__(self, ncat):
         self.ncat = ncat
         self.temperature = tf.Variable(1.0, trainable=False, name="gumbel_temp") # Example temperature

     def pd_from_flat(self, flat):
         """Creates a TFP RelaxedOneHotCategorical distribution."""
         # 'flat' are the logits
         return tfp.distributions.RelaxedOneHotCategorical(self.temperature, logits=flat)

     def param_shape(self): # Info only
         return [self.ncat]
     def sample_shape(self): # Info only
         return [self.ncat] # Outputs one-hot relaxed vector
     def sample_dtype(self): # Info only
         return tf.float32


class BernoulliPdType:
    def __init__(self, size):
        self.size = size

    def pd_from_flat(self, flat):
        """Creates a TFP Bernoulli distribution from logits."""
        # 'flat' are the logits
        return tfp.distributions.Bernoulli(logits=flat)

    def param_shape(self): # Info only
        return [self.size]
    def sample_shape(self): # Info only
        return [self.size]
    def sample_dtype(self): # Info only
        return tf.int32

# --- MultiCategorical ---
# TFP doesn't have a direct MultiCategorical. Can be implemented using Independent
# or handled by splitting/joining outside the distribution object.

class MultiCategoricalPdType:
    def __init__(self, ncats):
        """ncats: A list or tuple of the number of categories for each dimension."""
        self.ncats = ncats

    def pd_from_flat(self, flat):
        """
        Creates a collection of independent Categorical distributions.
        Assumes 'flat' is concatenated logits for all dimensions.
        """
        logits_list = tf.split(flat, self.ncats, axis=-1)
        dists = [tfp.distributions.Categorical(logits=lg) for lg in logits_list]
        # Using TFP's Independent wrapper combines them logically
        # return tfp.distributions.Independent(
        #     tfp.distributions.Categorical(logits=logits_list), # This expects logits shape [..., sum(ncats)], need reshape/split first
        #     reinterpreted_batch_ndims=1 # Treat the last dim (num components) as part of the event
        # )
        # Simpler approach: return the list of distributions and handle operations (sample, log_prob) by iterating
        return dists # Return a list of TFP Categorical distributions

    def sample(self, dists):
        """Sample from a list of TFP distributions."""
        return tf.stack([dist.sample() for dist in dists], axis=-1)

    def log_prob(self, dists, x):
        """Calculate log_prob for a list of TFP distributions."""
        # x shape should be [..., num_components]
        x_split = tf.unstack(x, axis=-1)
        log_probs = [dist.log_prob(x_i) for dist, x_i in zip(dists, x_split)]
        return tf.add_n(log_probs) # Sum log probs for independence

    def entropy(self, dists):
        """Calculate total entropy for a list of TFP distributions."""
        entropies = [dist.entropy() for dist in dists]
        return tf.add_n(entropies)

    def param_shape(self): # Info only
        return [sum(self.ncats)]
    def sample_shape(self): # Info only
        return [len(self.ncats)]
    def sample_dtype(self): # Info only
        return tf.int32


# --- Factory function ---
def make_pdtype(ac_space):
    """Creates a PdType object based on the Gym action space."""
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        # Use Categorical for discrete actions
        # return CategoricalPdType(ac_space.n)
        # Or SoftCategorical for Gumbel-Softmax (often used in MADDPG)
        return SoftCategoricalPdType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        # Use our custom list-based approach or find a suitable TFP equivalent
        # return MultiCategoricalPdType(ac_space.nvec) # nvec gives the list of category counts
         # If using Soft version:
         raise NotImplementedError("SoftMultiCategorical needs careful implementation with TFP.")
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPdType(ac_space.n)
    else:
        raise NotImplementedError(f"Action space type {type(ac_space)} not supported.")


# The original Pd class methods (logp, kl, entropy, sample, mode)
# are now replaced by the corresponding methods of the TFP distribution objects
# returned by pd_from_flat.
# Example:
# pdtype = DiagGaussianPdType(2)
# params = model(observation) # Get NN output [batch, 4]
# pd = pdtype.pd_from_flat(params) # pd is a tfp.distributions.MultivariateNormalDiag instance
# action_sample = pd.sample()
# log_prob = pd.log_prob(action_sample)
# entropy = pd.entropy()
# mode = pd.mode() # which is the mean for Gaussian