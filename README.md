# Ouro Model: Sequence Modeling Experiments

This repository contains JAX/Flax implementations of sequence models for a reversal task, specifically designed to investigate length generalization capabilities of standard and looped Transformers.

## Project Structure

- `src/`: Source code for models and training utilities.
  - `model.py`: Transformer implementations (Standard and Looped).
  - `train.py`: Training step and evaluation logic.
  - `utils.py`: Data generation (reversal task) and metrics.
- `main.py`: Entry point for running experiments.
- `requirements.txt`: Python dependencies.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ouro_model
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` includes `jax-metal` for macOS GPU acceleration. Adjust JAX installation for other platforms (CUDA/TPU) as needed.*

## Usage

Run the experiments using `main.py`.

```bash
python3 main.py
```

### Configuration

You can configure the experiment by modifying the variables at the top of the `main()` function in `main.py`.

**Key Parameters:**

- `model_type`: `'standard'` or `'looped'`
- `training_iterations`: Number of layers for StandardTransformer or iterations for LoopedTransformer.
- `inference_iterations_list`: List of iteration counts to test during inference (LoopedTransformer only).

**Example Configuration for Inference Scaling Experiment:**

```python
# Inference Scaling Experiment
model_type = 'looped'
training_iterations = 2
inference_iterations_list = [2, 6, 10, 20]
```

## Comparisons

The codebase supports three main experimental setups:

1.  **Standard Transformer**: `model_type='standard'`, `num_layers=2`.
    - Classic deep learning approach with independent weights per layer.
2.  **Looped Transformer**: `model_type='looped'`, `training_iterations=K`.
    - Weight-tied recurrent Transformer. Same block applied K times.
3.  **Inference Scaling**: Train `looped` with small K (e.g., 2), test with larger K (e.g., 6, 10, 20).
    - Investigates if additional compute at test time improves performance.

## Results

Experiments on the Reversal Task (Train Len: 3-10, Test Len: 20-50):

- **Generalization**: None of the tested configurations (Standard, Looped K=2, Looped K=6, Inference Scaling) successfully generalized to lengths > 10.
- **Inference Scaling**: Increasing iterations at test time for a model trained with K=2 resulted in determining performance (0% accuracy).

## License

[MIT License](LICENSE)
