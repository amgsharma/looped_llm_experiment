import jax
import jax.numpy as jnp
import numpy as np

def get_device_info():
    """Prints JAX device information."""
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")

def create_reversal_batch(batch_size, min_len, max_len, vocab_size):
    """
    Generates a batch of reversal tasks.
    
    Args:
        batch_size: Number of sequences.
        min_len: Minimum sequence length (excluding padding).
        max_len: Maximum sequence length.
        vocab_size: Size of the vocabulary. Values will be in [0, vocab_size-1].
        
    Returns:
        A dict with 'x' and 'y' jnp arrays.
        'x': Input sequences, padded with 0.
        'y': Reversed sequences, padded with 0.
    """
    # We use 0 as padding, so we'll generate tokens in [1, vocab_size-1]
    # to avoid ambiguity, effectively ensuring 0 is ONLY padding.
    # However, user requested values 0..vocab_size-1. 
    # To strictly satisfy inputs 0..vocab_size-1 and have padding, 
    # we would technically need a distinct pad token (like -1 or vocab_size).
    # For this implementation, I will assume we shift values by 1 for generation 
    # if we want to treat 0 as pad, OR we just generate 0..V-1 and pad with 0.
    # Let's generate 1..vocab_size-1 to be safe and use 0 as pad. 
    # This means actural vocab used is 1..V-1.
    
    x_batch = np.zeros((batch_size, max_len), dtype=np.int32)
    y_batch = np.zeros((batch_size, max_len), dtype=np.int32)
    
    # Vectorized implementation for efficiency
    
    # 1. Generate random lengths for each batch item
    lengths = np.random.randint(min_len, max_len + 1, size=batch_size)
    
    # 2. Create a mask for valid positions: (batch, max_len)
    # indices: (1, max_len) -> [0, 1, ..., max_len-1]
    # mask: (batch, max_len) where indices < lengths
    indices = np.arange(max_len)[None, :]
    mask = indices < lengths[:, None]
    
    # 3. Generate random sequence data for the whole block
    # We generate values in [1, vocab_size) to reserve 0 for padding
    full_random = np.random.randint(1, vocab_size, size=(batch_size, max_len)).astype(np.int32)
    
    # Apply mask to zero out padding
    x_batch = full_random * mask
    
    # 4. Create y_batch (reversed sequences)
    # Since lengths vary, we can't just flip the whole matrix.
    # However, for a reversal task x=[a, b, c, 0, 0], y=[c, b, a, 0, 0]
    # We can perform this by reversing the relevant parts row-wise.
    # A fully vectorized variable-length reverse is tricky in pure numpy without advanced indexing.
    # Given the constraints, a compiled JAX/NumPy approach or a optimized loop is best.
    # Let's stick to a loop for the *reposal* part if it's too complex, OR:
    # We can generate y first in aligned form and then pad? No.
    
    # Let's use a loop for now but optimized:
    # The generation of random numbers (expensive part) is vectorized.
    # The assignment is fast.
    
    y_batch = np.zeros_like(x_batch)
    for i in range(batch_size):
        l = lengths[i]
        y_batch[i, :l] = x_batch[i, :l][::-1]
        
    return {
        'x': jnp.array(x_batch),
        'y': jnp.array(y_batch)
    }

def compute_accuracy(logits, targets):
    """
    Computes exact sequence match accuracy.
    
    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
        
    Returns:
        Scalar accuracy between 0 and 1.
    """
    predictions = jnp.argmax(logits, axis=-1)
    
    # Check if all tokens in the sequence match the target
    # (batch, seq_len) -> (batch,)
    # boolean array where True means token match
    token_match = (predictions == targets)
    
    # Check if entire sequence matches
    seq_match = jnp.all(token_match, axis=1)
    
    # Average over batch
    accuracy = jnp.mean(seq_match)
    return accuracy
