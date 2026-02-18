import jax
import jax.numpy as jnp
import numpy as np
from src.model import StandardTransformer, LoopedTransformer
from src.train import create_train_state, train_step, eval_step
from src.utils import get_device_info, create_reversal_batch, compute_accuracy

def evaluate_on_length(state, seq_len, num_batches, vocab_size, batch_size=32, num_iterations=None):
    """
    Evaluates the model on a specific sequence length.
    """
    accuracies = []
    for _ in range(num_batches):
        batch = create_reversal_batch(batch_size, seq_len, seq_len, vocab_size)
        logits = eval_step(state, batch, num_iterations=num_iterations)
        acc = compute_accuracy(logits, batch['y'])
        accuracies.append(acc)
    return np.mean(accuracies)

def main():
    print("Initializing JAX Sequence Modeling Project...")
    get_device_info()
    
    # --- Configuration ---
    # Inference Scaling Experiment
    model_type = 'looped'
    training_iterations = 2
    inference_iterations_list = [2, 6, 10, 20]
    
    print(f"Configuration: Model={model_type}, Training K={training_iterations}")
    print(f"Inference Scaling: Testing K={inference_iterations_list}")

    # Hyperparameters
    vocab_size = 100
    hidden_size = 64
    learning_rate = 1e-3
    batch_size = 32
    min_len = 3
    max_len = 10
    num_steps = 5000
    
    # Initialize model
    # We only support LoopedTransformer for this experiment
    model = LoopedTransformer(
        vocab_size=vocab_size, 
        hidden_size=hidden_size, 
        num_iterations=training_iterations
    )

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    # Pass a dummy input with max_len for initialization
    dummy_input = jnp.ones([1, max_len], dtype=jnp.int32)
    params = model.init(init_rng, dummy_input)['params']
    
    state = create_train_state(init_rng, model, learning_rate, vocab_size, hidden_size)
    
    print("Starting reversal task training loop...")
    for step in range(num_steps):
        batch = create_reversal_batch(batch_size, min_len, max_len, vocab_size)
        state, loss, logits = train_step(state, batch)
        
        if step % 100 == 0:
            acc = compute_accuracy(logits, batch['y'])
            print(f"Step {step}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
            
    print("Training loop completed successfully!")
    
    print("\nStarting Evaluation Phase with Inference Scaling...")
    test_lengths = [10, 20, 30, 40, 50]
    num_eval_batches = 10
    
    for length in test_lengths:
        print(f"\nEvaluating Length: {length}")
        for inf_k in inference_iterations_list:
            avg_acc = evaluate_on_length(
                state, 
                length, 
                num_eval_batches, 
                vocab_size, 
                batch_size, 
                num_iterations=inf_k
            )
            print(f"  Inference K={inf_k}: Accuracy: {avg_acc:.4f}")

if __name__ == "__main__":
    main()
