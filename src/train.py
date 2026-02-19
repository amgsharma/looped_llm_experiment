import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm

def create_train_state(rng, model, learning_rate, vocab_size, hidden_size):
    """Creates initial `TrainState`."""
    params = model.init(rng, jnp.ones([1, 10], dtype=jnp.int32))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

def train_step(state, batch, num_iterations=None):
    """Trains on one batch."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['x'], num_iterations=num_iterations)
        # Calculate loss directly against the target sequence (batch['y']) using cross entropy
        # No shifting; we predict the entire sequence at once.
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['y']).mean()
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, logits

def eval_step(state, batch, num_iterations=None):
    """Evaluates on one batch (returns logits). 
       Optionally overrides num_iterations for LoopedTransformer.
    """
    if num_iterations is not None:
        logits = state.apply_fn({'params': state.params}, batch['x'], num_iterations=num_iterations)
    else:
        logits = state.apply_fn({'params': state.params}, batch['x'])
    return logits

# JIT compile with static_argnames
train_step = jax.jit(train_step, static_argnames=('num_iterations',))
eval_step = jax.jit(eval_step, static_argnames=('num_iterations',))
