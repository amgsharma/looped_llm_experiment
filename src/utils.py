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
    
    for i in range(batch_size):
        length = np.random.randint(min_len, max_len + 1)
        # generate random sequence of length 'length'
        # values in [1, vocab_size-1] to reserve 0 for padding
        seq = np.random.randint(1, vocab_size, size=(length,))
        
        # input: seq followed by 0s
        x_batch[i, :length] = seq
        
        # output: reversed seq followed by 0s
        y_batch[i, :length] = seq[::-1]
        
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
