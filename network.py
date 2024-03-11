import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv
'''
In this network, we first update node features through GTA for each frame, then generate edge features using the corresponding node features through MLP, then update edge features
through Transformer.
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerLayer(nn.Module):
    def __init__(self, feature_dim):
        super(TransformerLayer, self).__init__()
        self.positional_encoding = PositionalEncoding(feature_dim)
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=2, dropout=0.1)  # Use 1 attention head
        self.norm1 = nn.LayerNorm(feature_dim) # Layer Normalization normalizes inputs across the feature dimension
        self.norm2 = nn.LayerNorm(feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, 4 * feature_dim),
            nn.ReLU(),
            nn.Linear(4 * feature_dim, feature_dim)
        )

    def forward(self, x):
        # x shape: (seq_len, batch_size, feature_dim)
        # Apply positional encoding
        x = self.positional_encoding(x)
        # Multi-head self-attention
        attn_output, _ = self.attention(x, x, x) # attn_output:[seq_len, batch_size, embed_dim]
        x = self.norm1(x + attn_output)
        # Position-wise feed-forward networks
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x
    
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels = [1024, 1024, 1024], kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # Calculate the padding based on the kernel and dilation
            padding = ((kernel_size - 1) * dilation_size) // 2
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size,
                                 stride=1, padding=padding,
                                 dilation=dilation_size),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x needs to be of shape [batch_size, features, seq_length]
        x = x.permute(0, 2, 1)
        return self.network(x).permute(0, 2, 1)
    
class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATLayer, self).__init__()
        self.conv = GATConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Flatten batch and temporal dimensions for GATConv
        _, num_nodes, node_dim = x.shape
        x = x.contiguous().view(-1, x.size(-1))
        edge_index = edge_index.contiguous().view(2, -1)
        
        x = F.relu(self.conv(x, edge_index))
        
        # Restore batch and temporal dimensions
        x = x.view(-1, num_nodes, x.size(-1))
        #x = F.relu(self.conv(x, edge_index))
        return x
    
class EdgeNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(EdgeNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index):
         # Flatten batch and temporal dimensions for edge feature extraction
        x = x.view(-1, x.size(-1))
        edge_index = edge_index.contiguous().view(2, -1)
        
        start, end = edge_index
        edge_features = torch.cat([x[start], x[end]], dim=1)
        edge_features = self.net(edge_features)

        # Restore batch and temporal dimensions
        edge_features = edge_features.view(-1, edge_index.size(1), edge_features.size(-1))
        return edge_features

class GATUpdateAndEdgeTransform(nn.Module):
    def __init__(self, node_dim, edge_dim, num_layers, hidden_dim):
        super(GATUpdateAndEdgeTransform, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GATLayer(node_dim, node_dim))
        self.edge_network = EdgeNetwork(node_dim, hidden_dim, edge_dim)
        self.temporal_conv_net = TemporalConvNet(edge_dim)

    def forward(self, x, edge_index):
        batch_size, num_frames, num_nodes, node_dim = x.shape
        num_edges = edge_index.shape[-1]
        edge_dim = self.edge_network.net[-1].out_features

        # Reshape tensors to treat frames in a batch-like manner
        x_reshaped = x.view(-1, num_nodes, node_dim)
        edge_index_reshaped = edge_index.contiguous().view(-1, 2, edge_index.size(-1))

        # Generate edge features and update node features
        # edge_features_reshaped = self.edge_network(x_reshaped, edge_index_reshaped)
        for layer in self.layers:
            x_reshaped = layer(x_reshaped, edge_index_reshaped)

        # x = self.temporal_conv_net(x_reshaped)
        x = x.view(batch_size, num_frames, num_nodes, node_dim)

        # Reshape back to [batch_size, num_frames, ..., ...]
        #edge_features = edge_features_reshaped.view(batch_size, num_frames, -1, edge_dim)       
        
        # x = x_reshaped.view(batch_size, num_frames, num_nodes, node_dim)

        # Rearrange dimensions for transformer
        # edge_features_reshaped = edge_features.permute(1, 2, 0, 3).contiguous().view(num_frames, batch_size * num_edges, edge_dim)
        
        # Apply transformer
        # edge_features_transformed = self.transformer_layer(edge_features_reshaped)

        # Restore original dimensions
        # edge_features = edge_features_transformed.view(num_frames, num_edges, batch_size, edge_dim).permute(2, 0, 1, 3)
        edge_features = None
        return edge_features, x
    
class MultiLayerGATUpdateAndEdgeTransform(nn.Module):
    def __init__(self, node_dim, edge_dim, num_layers, hidden_dim, num_transform_layers):
        super(MultiLayerGATUpdateAndEdgeTransform, self).__init__()
        self.transform_layers = nn.ModuleList()
        for _ in range(num_transform_layers):
            self.transform_layers.append(GATUpdateAndEdgeTransform(node_dim, edge_dim, num_layers, hidden_dim))

    def forward(self, x, edge_index):
        edge_features_accumulated = None
        node_features_accumulated = None
        
        for layer in self.transform_layers:
            edge_features, x = layer(x, edge_index)
            
            if edge_features_accumulated is None and node_features_accumulated is None:
                edge_features_accumulated = edge_features
                node_features_accumulated = x
            else:
                # Residual addition
                # edge_features_accumulated = edge_features_accumulated + edge_features
                edge_features_accumulated = None
                node_features_accumulated = node_features_accumulated + x
        
        return edge_features_accumulated, node_features_accumulated

class NodeLocationEmbedding(nn.Module):
    def __init__(self,node_initial_dim, node_dim):
        super(NodeLocationEmbedding, self).__init__()
        self.layer1 = nn.Linear(node_initial_dim, 64)  # First hidden layer
        self.layer2 = nn.Linear(64, node_dim)  # Output layer

    def forward(self, x):
        # x shape: [frame_number, 52, 4]
        x = F.relu(self.layer1(x))  # shape: [frame_number, 52, 64]
        x = self.layer2(x)  # shape: [frame_number, 52, 128]
        return x
    
class NodeFeatureEmbedding(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(NodeFeatureEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, node_ids):
        return self.embedding(node_ids)
    
class NodeEmbedding(nn.Module):
    def __init__(self, node_initial_dim, node_dim):
        super(NodeEmbedding, self).__init__()
        self.node_location_embedding = NodeLocationEmbedding(node_initial_dim,node_dim)
        self.node_feature_embedding = NodeFeatureEmbedding(52, node_dim)
        self.embedding_fusion = nn.Sequential(
            nn.Linear(2*node_dim, node_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        batch_size, num_frames, num_nodes, _ = x.shape
        node_location_embedding = self.node_location_embedding(x)
        node_ids = torch.arange(0, num_nodes).to(x.device)
        node_feature_embedding = self.node_feature_embedding(node_ids)
        expanded_node_embeddings = node_feature_embedding.unsqueeze(0).unsqueeze(0).repeat(batch_size, num_frames, 1, 1)
        concatenated_tensor = torch.cat((node_location_embedding, expanded_node_embeddings), dim=3)
        return self.embedding_fusion(concatenated_tensor)   

class WeightedSumEdgeFeatures(nn.Module):
    def __init__(self, edge_dim):
        super(WeightedSumEdgeFeatures, self).__init__()
        self.attention_network = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, edge_features):
        # Assume edge_features has shape [batch_size, num_frames, num_edges, edge_dim]
        batch_size, num_frames, num_edges, edge_dim = edge_features.shape
        average_features = torch.mean(edge_features, dim=1)  # Shape: [batch_size, num_edges, edge_dim]
        
        attention_scores = self.attention_network(average_features)  # Shape: [batch_size, num_edges, 1]
        attention_scores = F.softmax(attention_scores, dim=1)  # Softmax over the num_edges dimension
        attention_scores = attention_scores.unsqueeze(1)  # Shape: [batch_size, 1, num_edges, 1]
        
        edge_features = torch.sum(edge_features * attention_scores, dim=1, keepdim=False)  # Sum over the num_frames dimension
        # Resulting shape: [batch_size, num_edges, edge_dim]
        return edge_features
    
class Flatten_nodes(nn.Module):
    def __init__(self, node_dim):
        super(Flatten_nodes, self).__init__()
        self.flatten_nodes = nn.Sequential(
            nn.Linear(node_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    def forward(self,x):
        x = self.flatten_nodes(x)
        return x.view(*x.shape[:-2], -1)

class Concatenate_i3d(nn.Module):
    def __init__(self):
        super(Concatenate_i3d, self).__init__()
        self.i3d_forward = nn.Linear(2048,1024)
        self.concatenate = nn.Linear(32*52 + 1024, 1024)
    def forward(self,x_flattened,i3d_features):
        i3d_features = self.i3d_forward(i3d_features)
        concatenated_features = torch.cat((x_flattened,i3d_features), dim=-1)
        return self.concatenate(concatenated_features)

class BimanualActionPredictionNetwork(nn.Module):
    def __init__(self,node_initial_dim, node_dim, edge_dim, num_layers, hidden_dim, num_transform_layers, num_classes):
        super(BimanualActionPredictionNetwork, self).__init__()
        self.node_embedding = NodeEmbedding(node_initial_dim,node_dim)
        self.gat_transform = MultiLayerGATUpdateAndEdgeTransform(node_dim, edge_dim, num_layers, hidden_dim, num_transform_layers)

        self.flatten_nodes = Flatten_nodes(node_dim)
        self.concatenate_features = Concatenate_i3d()

        self.temporal_conv_net = TemporalConvNet(1024)

        self.action_classifier_hand_0 = nn.Sequential(
            nn.Linear(1024, num_classes)  # Fully connected layer for action classification
        )

        self.action_classifier_hand_1 = nn.Sequential(
            nn.Linear(1024, num_classes)  # Fully connected layer for action classification
        )
        
    def forward(self, x, edge_index, i3d_features):
        #print('before embedding x.shape =', x.shape)
        batch_size, num_frames, num_nodes, node_dim = x.shape
        batch_size, num_frames, i3d_dim = i3d_features.shape
        x = self.node_embedding(x)
        #print('after embedding x.shape =', x.shape)
        half_edge_index = int(edge_index.shape[3]//2)
        hand_0_edge_index = edge_index[:,:,:,0:half_edge_index]
        hand_1_edge_index = edge_index[:,:,:,half_edge_index:]
        hand_0_edge_features, node_features_hand_0 = self.gat_transform(x,hand_0_edge_index)
        hand_1_edge_features, node_features_hand_1 = self.gat_transform(x,hand_1_edge_index)

        flattened_node_features_hand_0 = self.flatten_nodes(node_features_hand_0)
        features_hand_0 = self.concatenate_features(flattened_node_features_hand_0, i3d_features)

        flattened_node_features_hand_1 = self.flatten_nodes(node_features_hand_1)
        features_hand_1 = self.concatenate_features(flattened_node_features_hand_1, i3d_features)

        features_hand_0 = self.temporal_conv_net(features_hand_0)
        features_hand_1 = self.temporal_conv_net(features_hand_1)

        hand_0_action_class = self.action_classifier_hand_0(features_hand_0)
        hand_1_action_class = self.action_classifier_hand_1(features_hand_1)
        
        return hand_0_action_class, hand_1_action_class