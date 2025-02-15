import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class AdvancedTemporalGraphNetwork(nn.Module):
    def __init__(self, num_features, hidden_channels, num_nodes, dropout=0):
        super(AdvancedTemporalGraphNetwork, self).__init__()
        self.init_params = {
            'num_features': num_features,
            'hidden_channels': hidden_channels,
            'num_nodes': num_nodes,
            'dropout': dropout
        }
        self.feature_reducer = nn.Linear(num_features, hidden_channels)
        self.graph_conv1 = pyg_nn.SAGEConv(hidden_channels, hidden_channels)
        self.graph_conv2 = pyg_nn.SAGEConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=2,
            batch_first=True,
            dropout=dropout
        )

        # Prediction head now outputs only 1 value for pm2p5
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels // 2, 1)  # Output size set to 1
        )

    # Update your model's forward method to handle the sequence properly:
    def forward(self, x, edge_index, time_sequence):
        """
        Forward pass through the network
        Args:
          x: Input features tensor [batch_size, num_nodes, num_features]
          edge_index: Graph connectivity [2, num_edges]
          time_sequence: Temporal sequence data [batch_size, sequence_length, num_features]
        Returns:
          predictions: PM2.5 predictions [batch_size, 1]
        """
        num_features = self.init_params['num_features']
        num_nodes = self.init_params['num_nodes']
        batch_size = x.shape[0]
        if x is None:
          x = torch.randn(batch_size, num_nodes, num_features, device=edge_index.device)
        x = x.float()

        time_sequence = time_sequence.float()

        # Reshape the input tensor to match the expected shape
        x_flat = x.view(-1, num_features)  # Flatten the batch and nodes dimensions

        # Apply feature reduction and graph convolutions
        spatial_features = self.feature_reducer(x_flat)
        spatial_features = self.dropout(F.relu(self.graph_conv1(spatial_features, edge_index)))
        spatial_features = self.dropout(F.relu(self.graph_conv2(spatial_features, edge_index)))

        # Reshape back to sequence format
        spatial_features = spatial_features.view(batch_size, num_nodes, -1)

        # Process temporal features
        # Reshape and reduce temporal features
        time_flat = time_sequence.view(-1, num_features)
        temporal_features = self.feature_reducer(time_flat)
        temporal_features = temporal_features.view(batch_size, -1, temporal_features.size(-1))

        # Apply LSTM and attention
        lstm_out, _ = self.temporal_lstm(temporal_features)
        attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)

        # Combine spatial and temporal features
        # Take the mean of spatial features across nodes and add to temporal features
        combined_features = attn_out[:, -1, :] + spatial_features.mean(dim=1)

        # Generate final predictions
        predictions = self.prediction_head(combined_features)

        return predictions


print(f'Model Compiled')
