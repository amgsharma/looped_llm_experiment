import jax
import jax.numpy as jnp
import numpy as np
from src.model import StandardTransformer, LoopedTransformer
from src.train import create_train_state, train_step, eval_step
from src.utils import get_device_info, create_reversal_batch, compute_accuracy

def evaluate_on_length(state, seq_len, num_batches, vocab_size, batch_size=32):
    """
    Evaluates the model on a specific sequence length.
    """
    accuracies = []
    for _ in range(num_batches):
        batch = create_reversal_batch(batch_size, seq_len, seq_len, vocab_size)
        # We don't override num_iterations during standard evaluation here
        logits = eval_step(state, batch)
        acc = compute_accuracy(logits, batch['y'])
        accuracies.append(acc)
    return np.mean(accuracies)

def run_experiment(model_type, num_layers_or_iterations, experiment_name):
    print(f"\n=== {experiment_name} ===")
    
    # Hyperparameters for Final Demonstration
    vocab_size = 100
    hidden_size = 64
    learning_rate = 1e-3
    batch_size = 32
    min_len = 3
    max_len = 50 # Train up to 50
    num_steps = 8000 # Increased steps for longer sequences
    
    # Initialize model
    model = LoopedTransformer(
        vocab_size=vocab_size, 
        hidden_size=hidden_size, 
        num_iterations=num_layers_or_iterations
    )
    
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    # Initialization
    dummy_input = jnp.ones([1, max_len], dtype=jnp.int32)
    params = model.init(init_rng, dummy_input)['params']
    
    state = create_train_state(init_rng, model, learning_rate, vocab_size, hidden_size)
    
    print("Starting training loop...")
    final_train_acc = 0.0
    for step in range(num_steps):
        batch = create_reversal_batch(batch_size, min_len, max_len, vocab_size)
        
        state, loss, logits = train_step(state, batch)
        
        if step % 500 == 0:
            acc = compute_accuracy(logits, batch['y'])
            print(f"Step {step}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
            final_train_acc = acc
    
    # Final training accuracy check
    batch = create_reversal_batch(batch_size, min_len, max_len, vocab_size)
    logits = eval_step(state, batch)
    final_train_acc = compute_accuracy(logits, batch['y'])
            
    print("Training loop completed!")
    
    # Evaluation
    test_len = 50
    num_eval_batches = 10
    
    test_acc = evaluate_on_length(state, test_len, num_eval_batches, vocab_size, batch_size)
        
    # Print Results in requested format
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Test Accuracy (Len 50): {test_acc:.4f}")
    
    return test_acc

def main():
    print("Initializing JAX Sequence Modeling Project...")
    get_device_info()
    
    # Final Demonstration: LoopedTransformer (K=2) on 3-50
    run_experiment('looped', 2, 'LoopedTransformer (K=2)')

if __name__ == "__main__":
    main()
