import jax
import jax.numpy as jnp


def sample_t_discrete(flow_steps: int, rng, bs):
    t = jax.random.randint(rng, (bs,), minval=0, maxval=flow_steps)
    return t.astype(jnp.float32) / flow_steps


def sample_t_normal(rng, bs):
    t = jax.random.uniform(rng, (bs,), minval=0, maxval=1.0)
    return t.astype(jnp.float32)
