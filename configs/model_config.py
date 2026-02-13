"""Model configuration for the small GPT-2 variant (~10M parameters).

This is a deliberately small model so the full RLHF pipeline trains in minutes
on a single Apple M4 Metal GPU. The architecture mirrors GPT-2 but scaled down.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the GPT-2 style language model."""
    vocab_size: int = 50257       # GPT-2 tokenizer vocabulary size
    max_seq_len: int = 256        # Maximum sequence length (short for fast training)
    n_layers: int = 4             # Number of transformer blocks
    n_heads: int = 4              # Number of attention heads
    d_model: int = 256            # Hidden dimension
    d_ff: int = 1024              # Feed-forward intermediate dimension (4 * d_model)
    dropout_rate: float = 0.1     # Dropout rate (used during training)
    dtype: str = "float32"        # Default dtype; Metal does NOT support bfloat16

