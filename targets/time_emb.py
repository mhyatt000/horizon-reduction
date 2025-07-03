import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np

from math_utils import get_2d_sincos_pos_embed, modulate
from jax._src import core
from jax._src import dtypes
from jax._src.nn.initializers import _compute_fans

from jax._src import dtypes


def xavier_uniform_pytorchlike():
    def init(key, shape, dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        named_shape = core.as_named_shape(shape)
        if len(shape) == 2:  # Dense, [in, out]
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4:  # Conv, [k, k, in, out]. Assumes patch-embed style conv.
            fan_in = shape[0] * shape[1] * shape[2]
            fan_out = shape[3]
        else:
            raise ValueError(f'Invalid shape {shape}')

        variance = 2 / (fan_in + fan_out)
        scale = jnp.sqrt(3 * variance)
        param = jax.random.uniform(key, shape, dtype, -1) * scale

        return param

    return init


class TrainConfig:
    def __init__(self, dtype):
        self.dtype = dtype

    def kern_init(self, name='default', zero=False):
        if zero or 'bias' in name:
            return nn.initializers.constant(0)
        return xavier_uniform_pytorchlike()

    def default_config(self):
        return {
            'kernel_init': self.kern_init(),
            'bias_init': self.kern_init('bias', zero=True),
            'dtype': self.dtype,
        }


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    hidden_size: int
    tc: TrainConfig
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.normal(0.02),
            bias_init=self.tc.kern_init('time_bias'),
            dtype=self.tc.dtype,
        )(x)
        x = nn.silu(x)
        x = nn.Dense(
            self.hidden_size, kernel_init=nn.initializers.normal(0.02), bias_init=self.tc.kern_init('time_bias')
        )(x)
        return x

    # t is between [0, 1].
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                            These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = jax.lax.convert_element_type(t, jnp.float32)
        # t = t * max_period
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        embedding = embedding.astype(self.tc.dtype)
        return embedding


if __name__ == '__main__':
    from rich.pretty import pprint

    dtype = jnp.bfloat16
    tc = TrainConfig(dtype=dtype)
    model = TimestepEmbedder(hidden_size=16, tc=tc)
    rng = jax.random.PRNGKey(0)
    # tabulate_fn = flax.linen.tabulate(model_def, jax.random.PRNGKey(0))
    # times = jnp.array([1e-5, 1, 2, 3, 4, 5], dtype=dtype) / 128
    # times = jnp.linspace(1e-5, 128, 128, dtype=dtype) / 128
    t = jax.random.randint(rng, (128,), minval=0, maxval=128).astype(jnp.float32) / 128
    null = jnp.zeros((1,), dtype=dtype)

    params = model.init(rng, t)  # ["params"]

    embeddings = model.apply(params, t)

    pprint(np.array(embeddings).round(3))
    pprint(np.array(model.apply(params, null).round(3)))
