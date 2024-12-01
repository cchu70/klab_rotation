import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Asked Claude 3.5 Sonnet to convert the original code to pytorch (https://github.com/coxlab/prednet/blob/master/prednet.py)

This code uses an older version of keras
'''

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell for PredNet.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Combined gates for efficiency (input, forget, output, gate)
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, input_tensor, state):
        h_cur, c_cur = state #?
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Compute gates
        gates = self.conv(combined)
        ingate, forgetgate, cellgate, outgate = torch.split(gates, self.hidden_channels, dim=1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        c_next = (forgetgate * c_cur) + (ingate * cellgate) # long term?
        h_next = outgate * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        device = next(self.parameters()).device
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=device))


class PredNetLayer(nn.Module):
    """
    Single layer of PredNet consisting of error computation, ConvLSTM, and prediction units.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(PredNetLayer, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        # Error computation units (ON and OFF states)
        self.error_channels = input_channels * 2
        
        # Prediction unit (ConvLSTM)
        self.conv_lstm = ConvLSTMCell(
            self.error_channels + hidden_channels,  # Error units + feedback
            hidden_channels,
            kernel_size
        )
        
        # Prediction to input space
        self.conv_pred = nn.Conv2d(
            hidden_channels,
            input_channels,
            kernel_size=1
        )

    def forward(self, input_tensor, state, top_down=None):
        batch_size, _, height, width = input_tensor.shape
        h_prev, c_prev = state
        
        # Make prediction
        pred = self.conv_pred(h_prev)
        
        # Compute errors (ON and OFF states)
        pos_error = F.relu(input_tensor - pred)
        neg_error = F.relu(pred - input_tensor)
        error = torch.cat([pos_error, neg_error], dim=1)
        
        # Combine error with top-down feedback if available
        if top_down is not None:
            lstm_input = torch.cat([error, top_down], dim=1)
        else:
            lstm_input = error
            
        # Update ConvLSTM state
        h_cur, c_cur = self.conv_lstm(lstm_input, (h_prev, c_prev))
        
        return pred, error, (h_cur, c_cur)

    def init_state(self, batch_size, height, width):
        return self.conv_lstm.init_hidden(batch_size, height, width)


class PredNet(nn.Module):
    """
    Complete PredNet architecture for video sequence prediction.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        """
        Args:
            input_channels (list): Number of input channels for each layer
            hidden_channels (list): Number of hidden channels for each layer
            kernel_size (int): Kernel size for convolutions
        """
        super(PredNet, self).__init__()
        
        self.num_layers = len(input_channels)
        assert len(hidden_channels) == self.num_layers, "Mismatch in layer specifications"
        
        # Create layers
        self.layers = nn.ModuleList([
            PredNetLayer(input_channels[i], hidden_channels[i], kernel_size)
            for i in range(self.num_layers)
        ])
        
        # Upsample convolutions for top-down connections
        self.upsample_convs = nn.ModuleList([
            nn.Conv2d(hidden_channels[i+1], hidden_channels[i], kernel_size=1)
            for i in range(self.num_layers-1)
        ])
        
        # Pooling for bottom-up connections
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input_sequence, states=None):
        """
        Process a sequence of frames.
        
        Args:
            input_sequence: Tensor of shape (batch, time, channels, height, width)
            states: Initial states for all layers (optional)
            
        Returns:
            predictions: List of predictions for each timestep
            errors: List of errors for each layer and timestep
        """
        batch_size, seq_length, _, height, width = input_sequence.shape
        
        # Initialize states if not provided
        if states is None:
            states = self.init_states(batch_size, height, width)
            
        predictions = []
        all_errors = []
        
        # Process each timestep
        for t in range(seq_length):
            current_input = input_sequence[:, t]
            layer_errors = []
            
            # Forward pass through each layer
            inputs = [current_input]
            new_states = []
            
            # Bottom-up pass
            for l, layer in enumerate(self.layers):
                # Get top-down input if not at top layer
                top_down = None
                if l < self.num_layers - 1:
                    top_down = F.interpolate(
                        states[l+1][0],
                        size=(inputs[l].shape[2], inputs[l].shape[3]),
                        mode='nearest'
                    )
                    top_down = self.upsample_convs[l](top_down)
                
                # Process layer
                pred, error, new_state = layer(inputs[l], states[l], top_down)
                new_states.append(new_state)
                layer_errors.append(error)
                
                # Prepare input for next layer (if not at top)
                if l < self.num_layers - 1:
                    inputs.append(self.pool(error))
            
            # Update states
            states = new_states
            
            # Store predictions and errors
            predictions.append(pred)
            all_errors.append(layer_errors)
        
        # Stack predictions and reshape errors
        predictions = torch.stack(predictions, dim=1)
        errors = [torch.stack([all_errors[t][l] for t in range(seq_length)], dim=1)
                 for l in range(self.num_layers)]
        
        return predictions, errors

    def init_states(self, batch_size, height, width):
        """Initialize states for all layers."""
        states = []
        current_height, current_width = height, width
        
        for layer in self.layers:
            states.append(layer.init_state(batch_size, current_height, current_width))
            current_height //= 2
            current_width //= 2
            
        return states


# Example usage
def create_prednet(input_shape):
    """
    Create a PredNet model with example parameters.
    
    Args:
        input_shape: Tuple of (channels, height, width)
    """
    # Layer specifications
    input_channels = [input_shape[0], 32, 64, 128]
    hidden_channels = [32, 64, 128, 256]
    
    model = PredNet(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        kernel_size=3
    )
    
    return model


def train_step(model, optimizer, input_sequence):
    """
    Single training step for PredNet.
    
    Args:
        model: PredNet model
        optimizer: PyTorch optimizer
        input_sequence: Tensor of shape (batch, time, channels, height, width)
    """
    optimizer.zero_grad()
    
    # Forward pass
    predictions, errors = model(input_sequence)
    
    # Compute loss (prediction error at each layer)
    loss = 0
    for error in errors:
        loss = loss + error.pow(2).mean()
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


# Example training loop
def train_prednet(model, train_loader, num_epochs, device='cuda'):
    """
    Train PredNet model.
    
    Args:
        model: PredNet model
        train_loader: DataLoader for training sequences
        num_epochs: Number of training epochs
        device: Device to train on
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    avg_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, sequences in enumerate(train_loader):
            sequences = sequences.to(device)
            loss = train_step(model, optimizer, sequences)
            total_loss += loss
            
        avg_loss = total_loss / len(train_loader)
        avg_losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')