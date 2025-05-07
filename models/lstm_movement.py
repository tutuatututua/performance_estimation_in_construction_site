import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from typing import Optional, Union

class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for learning movement sequence representations
    and detecting anomalies based on reconstruction error.
    """
    def __init__(
        self,
        input_size: int,         # Number of joints/features (e.g., 6)
        sequence_length: int,    # Standardized sequence length (e.g., 80)
        embedding_dim: int,      # Size of the compressed representation
        hidden_size: int = 128,  # LSTM hidden size
        num_layers: int = 1,     # Number of LSTM layers (can be > 1)
        dropout: float = 0.2,
        bidirectional_encoder: bool = False, # Option for encoder
        conditional: bool = False, # Conditional Autoencoder flag
        num_job_categories: int = 3 # Only if conditional=True
    ):
        super(LSTMAutoencoder, self).__init__()

        self.sequence_length = sequence_length
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conditional = conditional
        self.num_job_categories = num_job_categories
        self.bidirectional_encoder = bidirectional_encoder

        encoder_output_factor = 2 if bidirectional_encoder else 1

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional_encoder
        )

        # Optional: Linear layer to get desired embedding dimension
        # If bidirectional, hidden_size * 2 -> embedding_dim
        self.fc_encode = nn.Linear(hidden_size * encoder_output_factor, embedding_dim)
        self.relu = nn.ReLU()

        # Conditional Embedding (if conditional)
        if self.conditional:
            self.job_embedding = nn.Embedding(num_job_categories, embedding_dim)
            # Add a learnable default job embedding for fallback
            self.default_job_embedding = nn.Parameter(torch.zeros(1, embedding_dim))
            nn.init.normal_(self.default_job_embedding, mean=0.0, std=0.02)
            # Adjust decoder input size to include conditional info
            decoder_input_size = embedding_dim * 2  # Concatenate sequence embedding and job embedding
        else:
            decoder_input_size = embedding_dim

        # Linear projections for encoder states to decoder initial states
        self.fc_h = nn.Linear(hidden_size * encoder_output_factor, hidden_size)
        self.fc_c = nn.Linear(hidden_size * encoder_output_factor, hidden_size)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=decoder_input_size,  # Adjusted decoder input size
            hidden_size=hidden_size,        # Match encoder's hidden size
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
            # Decoder is typically not bidirectional
        )

        # Output layer to reconstruct the original sequence feature dimension
        self.fc_decode = nn.Linear(hidden_size, input_size)

        # Regularization: Layer Normalization for better gradient flow
        self.layer_norm = nn.LayerNorm(hidden_size)

        logger.info(f"LSTMAutoencoder initialized: input_size={input_size}, seq_len={sequence_length}, embedding_dim={embedding_dim}, hidden={hidden_size}, conditional={conditional}, bidirectional_encoder={bidirectional_encoder}")

    def forward(self, x: torch.Tensor, job_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: Encode the sequence and then decode it to reconstruct.

        Args:
            x (torch.Tensor): Input sequence tensor (batch_size, sequence_length, input_size)
            job_ids (torch.Tensor, optional): Job type IDs for Conditional AE (batch_size). Defaults to None.

        Returns:
            torch.Tensor: Reconstructed sequence tensor (batch_size, sequence_length, input_size)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)  # Use actual sequence length from input

        # --- Encoding ---
        encoder_outputs, (hidden, cell) = self.encoder(x)

        # Get the final hidden state(s)
        if self.bidirectional_encoder:
            # Concatenate forward and backward final hidden states
            last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            last_cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)
        else:
            # Just use final hidden state
            last_hidden = hidden[-1, :, :]  # Shape: (batch, hidden_size)
            last_cell = cell[-1, :, :]      # Shape: (batch, hidden_size)

        # Create embedding
        embedding = self.relu(self.fc_encode(last_hidden))  # Shape: (batch, embedding_dim)

        # --- Conditional Encoding (if conditional) ---
        if self.conditional:
            # Fix: Graceful fallback for missing job_ids
            if job_ids is None or job_ids.size(0) != batch_size:
                logger.warning("Job IDs not provided for Conditional Autoencoder. Using default embedding.")
                # Create a default job embedding
                job_embedding = self.default_job_embedding.repeat(batch_size, 1)
            else:
                job_embedding = self.job_embedding(job_ids)  # Shape: (batch, embedding_dim)
                
            # Concatenate sequence embedding and job embedding
            decoder_embedding = torch.cat((embedding, job_embedding), dim=1) # Shape: (batch, embedding_dim * 2)
        else:
            decoder_embedding = embedding

        # --- Decoding ---
        # Prepare decoder input (repeat embedding)
        decoder_input = decoder_embedding.unsqueeze(1).repeat(1, seq_len, 1) 

        # IMPROVED: Transform encoder final states for decoder init states
        # Project the encoder final states to decoder initial states
        decoder_hidden_init = self.fc_h(last_hidden)
        decoder_cell_init = self.fc_c(last_cell)
        
        # Apply layer normalization for better gradient flow
        decoder_hidden_init = self.layer_norm(decoder_hidden_init)
        
        # Reshape to (num_layers, batch_size, hidden_size) for LSTM
        decoder_hidden_init = decoder_hidden_init.unsqueeze(0).repeat(self.num_layers, 1, 1)
        decoder_cell_init = decoder_cell_init.unsqueeze(0).repeat(self.num_layers, 1, 1)

        # Run decoder with transformed initial states
        decoder_outputs, _ = self.decoder_lstm(
            decoder_input, 
            (decoder_hidden_init, decoder_cell_init)
        )
        # decoder_outputs shape: (batch, sequence_length, hidden_size)

        # Reconstruct the original input feature dimension
        reconstructed_sequence = self.fc_decode(decoder_outputs)
        # reconstructed_sequence shape: (batch, sequence_length, input_size)

        return reconstructed_sequence