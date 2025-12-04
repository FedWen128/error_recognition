import torch
from torch import nn

from core.models.blocks import fetch_input_dim, MLP


class LSTMModel(nn.Module):
    """
    LSTM-based baseline model for error recognition.
    
    This model uses a bidirectional LSTM to process temporal sequences
    of features extracted from videos. It's designed as a baseline to 
    compare against the Transformer-based ErFormer model.
    
    Architecture:
    - Bidirectional LSTM (2 layers) encoder
    - MLP decoder (same as ErFormer)
    - Binary classification output
    
    Args:
        config: Configuration object containing model parameters
        hidden_size: Hidden dimension of LSTM (default: 256)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.3)
    """
    
    def __init__(self, config, hidden_size=256, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Fetch input dimension based on backbone and modality
        input_size = fetch_input_dim(config)
        
        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # MLP decoder (same as Transformer variant for consistency)
        # Bidirectional LSTM outputs hidden_size * 2
        decoder_input_dim = hidden_size * 2  # 512
        self.decoder = MLP(decoder_input_dim, 512, 1)
    
    def forward(self, x):
        """
        Forward pass of the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
               or (batch_size, input_size) for single timestep
        
        Returns:
            output: Tensor of shape (batch_size, 1) with logits for binary classification
        """
        # Handle NaN values in input
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # If input is 2D (batch_size, input_size), add sequence dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_len, hidden_size * 2)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output of the sequence
        last_output = lstm_out[:, -1, :]
        
        # Pass through MLP decoder (same as Transformer)
        output = self.decoder(last_output)
        
        return output
