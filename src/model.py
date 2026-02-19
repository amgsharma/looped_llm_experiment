import jax
import jax.numpy as jnp
from flax import linen as nn

class TransformerBlock(nn.Module):
    hidden_size: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        # --- Residual + LayerNorm (Attention) ---
        residual = x
        
        # Multi-head self-attention (full attention, no causal masking)
        # Default is full attention if no mask provided
        x = nn.SelfAttention(num_heads=self.num_heads)(x)
        
        # Add residual and normalize
        x = nn.LayerNorm()(residual + x)
        
        # --- Residual + LayerNorm (MLP) ---
        residual = x
        
        # MLP: hidden -> 4*hidden -> hidden
        x = nn.Dense(features=self.hidden_size * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_size)(x)
        
        # Add residual and normalize
        x = nn.LayerNorm()(residual + x)
        return x

class BaseTransformer(nn.Module):
    vocab_size: int
    hidden_size: int
    
    def add_positional_encoding(self, x):
        seq_len = x.shape[1]
        
        # 1. Token embedding
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_size)(x)
        
        # 2. Sinusoidal positional embeddings
        # Create position indices [0, 1, ..., seq_len-1]
        pos = jnp.arange(seq_len)[:, None]  # (seq_len, 1)
        
        # Compute div_term for sinusoidal encoding
        # exp(arange(0, d, 2) * -(log(10000.0) / d))
        div_term = jnp.exp(jnp.arange(0, self.hidden_size, 2) * -(jnp.log(10000.0) / self.hidden_size))
        
        # Calculate PE
        # (seq_len, hidden_size//2)
        pe_sin = jnp.sin(pos * div_term)
        pe_cos = jnp.cos(pos * div_term)
        
        # Interleave sin and cos
        # Create full PE matrix (seq_len, hidden_size)
        pe = jnp.zeros((seq_len, self.hidden_size))
        pe = pe.at[:, 0::2].set(pe_sin)
        pe = pe.at[:, 1::2].set(pe_cos)
        
        # Add to token embeddings (broadcast across batch)
        x = x + pe
        return x
    
    def project_to_vocab(self, x):
         # 4. Final linear projection to vocab size
        logits = nn.Dense(features=self.vocab_size)(x)
        return logits


class StandardTransformer(BaseTransformer):
    num_heads: int = 4
    num_layers: int = 2

    @nn.compact
    def __call__(self, x, num_iterations=None):
        # We ignore num_iterations for StandardTransformer
        x = self.add_positional_encoding(x)
        
        # Independent blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(hidden_size=self.hidden_size, num_heads=self.num_heads)(x)
            
        return self.project_to_vocab(x)

class LoopedTransformer(BaseTransformer):
    num_heads: int = 4
    num_iterations: int = 2

    @nn.compact
    def __call__(self, x, num_iterations=None):
        x = self.add_positional_encoding(x)
        
        # Shared block
        block = TransformerBlock(hidden_size=self.hidden_size, num_heads=self.num_heads)
        
        # Loop K times using the SAME block instance (shared weights)
        # Use provided num_iterations if available, else default to self.num_iterations
        iterations = num_iterations if num_iterations is not None else self.num_iterations
        
        # We need to use python control flow here since iterations can be dynamic/Python integer
        for _ in range(iterations):
            x = block(x)
            
        return self.project_to_vocab(x)

# Alias SequenceModel to StandardTransformer for backward compatibility if needed, 
# or just for clean migration.
SequenceModel = StandardTransformer
