# Multi-Head Attention in PyTorch

This repository implements a simple **Multi-Head Attention Mechanism** in PyTorch. It supports masking for use cases such as padding token exclusion and causal attention, making it suitable for various NLP and sequence-based tasks.

---

## Features
- **Multi-Head Attention**: Implementation of the core component of transformer-based models.
- **Masking Support**: Includes masking for padding and causal attention.
- **Scalable Design**: Configurable number of heads and embedding dimensions.
- **Reproducible Results**: Controlled parameter initialization with fixed seeds.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/multi-head-attention-pytorch.git
2. Navigate to the directory:
   ```bash
   cd multi-head-attention-pytorch
3. Install dependencies:
   ```bash
    pip install torch

## Usage
### Initialisation
import torch
from mhma import MHMA

# Define model parameters
embed_dim = 128  # Embedding dimension
heads = 8        # Number of attention heads
dq = 16          # Query dimension per head
dk = 16          # Key dimension per head
dv = 16          # Value dimension per head

## Forward Pass
### Without Mask
output = model(x)
print(output.shape)  # Expected: (32, 10, 128)

### With Mask
# Example mask
seq_len = 10
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(1)  # Causal mask
output = model(x, mask=mask)


## Implementation Details
The MHMA class implements the following:

- Query, Key, Value Matrices: Learned projections for multi-head attention.
- Attention Scores: Scaled dot-product attention computation.
- Masking: Optional masking to exclude certain positions.
- Output Projections: Combination of attention outputs from multiple heads.

## File Structure
multi-head-attention-pytorch/
- │
- ├── mhma.py             # Core Multi-Head Attention implementation
- ├── README.md           # Project documentation
- └── requirements.txt    # Dependencies (if any)


## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

