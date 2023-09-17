import jax
import jax.numpy as jnp
import haiku as hk

from jaxpm.painting import cic_read
from typing import Tuple

def _deBoorVectorized(x, t, c, p):
    """
    Evaluates S(x).

    Args
    ----
    x: position
    t: array of knot positions, needs to be padded as described above
    c: array of control points
    p: degree of B-spline
    """
    k = jnp.digitize(x, t) -1
    
    d = [c[j + k - p] for j in range(0, p+1)]
    for r in range(1, p+1):
        for j in range(p, r-1, -1):
            alpha = (x - t[j+k-p]) / (t[j+1+k-r] - t[j+k-p])
            d[j] = (1.0 - alpha) * d[j-1] + alpha * d[j]
    return d[p]


class NeuralSplineFourierFilter(hk.Module):
  """A rotationally invariant filter parameterized by 
  a b-spline with parameters specified by a small NN."""

  def __init__(self, n_knots=8, latent_size=16, name=None):
    """
    n_knots: number of control points for the spline  
    """
    super().__init__(name=name)
    self.n_knots = n_knots
    self.latent_size = latent_size

  def __call__(self, x, a):
    """ 
    x: array, scale, normalized to fftfreq default
    a: scalar, scale factor
    """

    net = jnp.sin(hk.Linear(self.latent_size)(jnp.atleast_1d(a)))
    net = jnp.sin(hk.Linear(self.latent_size)(net))

    w = hk.Linear(self.n_knots+1)(net) 
    k = hk.Linear(self.n_knots-1)(net)
    
    # make sure the knots sum to 1 and are in the interval 0,1
    k = jnp.concatenate([jnp.zeros((1,)),
                        jnp.cumsum(jax.nn.softmax(k))])

    w = jnp.concatenate([jnp.zeros((1,)),
                         w])

    # Augment with repeating points
    ak = jnp.concatenate([jnp.zeros((3,)), k, jnp.ones((3,))])

    return _deBoorVectorized(jnp.clip(x/jnp.sqrt(3), 0, 1-1e-4), ak, w, 3)


class Rescale(hk.Module):
  def __init__(self, n_input):
    super().__init__()  
    self.scale = hk.get_parameter("scale", [n_input], init=jnp.ones)
    self.bias = hk.get_parameter("bias", [n_input], init=jnp.zeros)

  def __call__(self, x):
    return self.scale * x + self.bias

class ConvBlock(hk.Module):
    def __init__(
        self,
        output_channels: int,
        kernel_shape: Tuple[int] = (3, 3, 3),
        padding: str = "SAME",
        batch_norm: bool = False,
        activation=jax.nn.relu,
    ):
        super().__init__()
        self.conv = hk.Conv3D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            padding=padding,
        )
        if batch_norm:
            self.batch_norm = hk.BatchNorm(decay_rate=0.9, epsilon=1e-5)
        else:
            self.batch_norm = None
        self.activation = activation

    def __call__(self, x):
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x

class FullyConnectedBlock(hk.Module):
    def __init__(
        self,
        hidden_dim: int, 
        activation=jax.nn.relu,
    ):
        super().__init__()
        self.fc = hk.Linear(hidden_dim)
        self.activation = activation
    
    def __call__(self, x):
        return self.activation(self.fc(x))

class CNN(hk.Module):
    def __init__(
        self,
        channels_hidden_dim: int,
        n_convolutions: int,
        n_fully_connected: int,
        n_globals_embedding: int = 2,
        globals_embedding_dim: int = 8,
        input_dim: int = 2,
        output_dim: int = 1,
        kernel_size: int = 3,
        pad_periodic: bool = True,
        embed_globals: bool = True, 
    ):
        super().__init__(name="CNN")
        self.kernel_size = kernel_size
        self.n_convolutions = n_convolutions
        self.pad_periodic = pad_periodic
        self.embed_globals = embed_globals
        kernel_shape = (self.kernel_size, self.kernel_size, self.kernel_size)
        self.receptive_field = self.kernel_size * self.n_convolutions
        self.learned_norm = Rescale(input_dim)
        self.conv_block = hk.Sequential(
            [
                ConvBlock(output_channels=channels_hidden_dim, kernel_shape=kernel_shape,)
                for _ in range(n_convolutions)
            ]
        )
        self.read_featues_at_pos = jax.vmap(
            cic_read,
            in_axes=(-1, None),
        )
        fcn_layers = [FullyConnectedBlock(channels_hidden_dim) for _ in range(n_fully_connected)] + [
            hk.Linear(output_dim)
        ]
        self.fcn_block = hk.Sequential(
            fcn_layers,
        )
        if self.embed_globals:
            self.globals_fcn = hk.Sequential(
                [FullyConnectedBlock(globals_embedding_dim) for _ in range(n_globals_embedding)] 
            )


    def concatenate_globals(self, features_at_pos, global_features):
        if global_features.ndim < 2:
            global_features = jnp.expand_dims(global_features, axis=-1)
        if self.embed_globals:
            global_features = self.globals_fcn(global_features)
        broadcast_globals = jnp.broadcast_to(
            global_features,
            (
                features_at_pos.shape[0],
                global_features.shape[-1],
            ),
        )
        return jnp.concatenate([features_at_pos, broadcast_globals], axis=-1)

    def __call__(
        self,
        x,
        positions,
        global_features=None,
    ):
        if positions.ndim == 1:
            positions = positions[None,...]
        x = self.learned_norm(x)  # [LR, LR, LR, input_dim]
        # wrap x for periodic boundary conditions
        if self.pad_periodic:
            x = jnp.pad(x, pad_width=self.receptive_field,mode='wrap')
        x = self.conv_block(x)  # [LR, LR, LR, n_channels_hidden]
        if self.pad_periodic:
            x = x[...,self.receptive_field:-self.receptive_field, self.receptive_field:-self.receptive_field, self.receptive_field:-self.receptive_field,:]
        # swap axes to make the last axis the feature axis for the linear layers
        features_at_pos = self.read_featues_at_pos(
            x, positions
        ).swapaxes(-2, -1)
        # [n_particles, n_channels_hidden]
        # Add time as a feature
        if global_features is not None:
            features_at_pos = self.concatenate_globals(
                features_at_pos, 
                global_features,
            )
            # [n_particles, n_channels_hidden + 1]
        features_at_pos = self.fcn_block(features_at_pos)
        # [n_particles, output_dim]
        return features_at_pos
