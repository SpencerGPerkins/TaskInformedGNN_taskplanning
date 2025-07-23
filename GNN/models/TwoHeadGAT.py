import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GATConv, NNConv, global_mean_pool        

class TwoHeadGAT(nn.Module):
    def __init__(self, in_dim, edge_feat_dim, hidden_dim, num_actions, heads=4, dropout=0.3):
        super(TwoHeadGAT, self).__init__()
        
        # MLP (Edge Embeddings)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, in_dim * in_dim)    
        )
        
        self.edge_embeddings = NNConv(in_dim, in_dim, self.edge_encoder, aggr="mean")
        
        # GAT Layers (Node Embeddings)
        self.gat_conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=False)
        self.gat_conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=False)
        
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Prediction Heads
        self.wire_head = nn.Linear(hidden_dim, 1)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions)
        )
        
    def forward(self, x, wire_mask, edge_index, edge_attr, batch):
        # Edge Embeddings
        x = self.edge_embeddings(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # Node Embeddings
        x1 = self.gat_conv1(x, edge_index)
        x1 = F.relu(x1)
        x2 = self.gat_conv2(x1, edge_index)
        x2 = x2 + x1 # Residual
        x2 = F.relu(x2)
        
        # Predictions
        p_wire = self.wire_head(x2[wire_mask]).squeeze(-1)
        
        # Graph level aggregation
        x_pooled = global_mean_pool(x2, batch)
        p_action = self.action_head(x_pooled)
        
        return p_wire, p_action
        
        