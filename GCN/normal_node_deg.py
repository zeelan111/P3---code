import torch
from torch_geometric.utils import degree, negative_sampling, subgraph
from torch_geometric.nn.models import GAE
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Load the .pt file with PyTorch
pt_file_path = r"/ceph/home/student.aau.dk/ca48dd/GCN/reduced_graph800.pt"
pyg_data = torch.load(pt_file_path)
print(f"Loaded PyTorch Geometric data with {pyg_data.num_nodes} nodes and {pyg_data.num_edges} edges.")

# Step 1: Validate and preprocess the graph
print("Filtering edge_index to ensure valid nodes...")
valid_mask = (pyg_data.edge_index[0] < pyg_data.num_nodes) & (pyg_data.edge_index[1] < pyg_data.num_nodes)
pyg_data.edge_index = pyg_data.edge_index[:, valid_mask]

# Debugging prints after filtering
print(f"After filtering - Max node ID: {pyg_data.edge_index.max().item()}")
print(f"After filtering - Edge Index Shape: {pyg_data.edge_index.shape}")

# Step 2: Compute Node Degrees and Normalize
print("Computing node degrees...")
node_degrees = degree(pyg_data.edge_index[0], num_nodes=pyg_data.num_nodes)  # Degree of each node

# Normalize the degree values to range [0, 1] using Min-Max Scaling
min_degree = torch.min(node_degrees)
max_degree = torch.max(node_degrees)
normalized_degrees = (node_degrees - min_degree) / (max_degree - min_degree)

# Assign normalized degrees as node features
pyg_data.x = normalized_degrees.unsqueeze(1)  # Add a dimension for feature matrix
print(f"Assigned normalized degree features: {pyg_data.x.shape}")

# Step 3: Define the GCN encoder
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Step 4: Initialize the GCN encoder and GAE model
in_channels = pyg_data.x.size(1)  # Input feature size
out_channels = 32  # Size of the output embeddings
encoder = GCN(in_channels, out_channels)
model = GAE(encoder)

# Step 5: Move the model and graph data to the appropriate device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
pyg_data = pyg_data.to(device)

# Step 6: Encode the graph
z = model.encode(pyg_data.x, pyg_data.edge_index)
print(f"Encoded graph embeddings shape: {z.shape}")

# Step 7: Generate negative samples for training
pos_edge_index = pyg_data.edge_index
neg_edge_index = negative_sampling(
    edge_index=pos_edge_index,
    num_nodes=z.size(0),
    num_neg_samples=pos_edge_index.size(1)
)

# Step 8: Compute the loss for training
loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
print(f"Reconstruction loss: {loss.item()}")

# Step 9: Validation with AUC-PR and AP using PyTorch
with torch.no_grad():
    # Compute predictions for positive and negative edges
    pos_pred = model.decode(z, pos_edge_index)
    neg_pred = model.decode(z, neg_edge_index)
    
    # Create labels for AUC-PR and AP calculation
    y_true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))])
    y_scores = torch.cat([pos_pred, neg_pred])
    
    # Sort scores and corresponding labels
    sorted_indices = torch.argsort(y_scores, descending=True)
    sorted_labels = y_true[sorted_indices]
    
    # Compute precision and recall
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1 - sorted_labels, dim=0)
    fn = tp[-1] - tp
    precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
    recall = tp / (tp + fn + 1e-8)
    
    # Compute AUC-PR using the trapezoidal rule
    auc_pr = torch.trapz(precision, recall)
    
    # Compute Average Precision (AP)
    ap = torch.sum((recall[1:] - recall[:-1]) * precision[1:])  # AP from PR curve
    
    print(f"Validation AUC-PR: {auc_pr.item():.4f}")
    print(f"Validation AP: {ap.item():.4f}")
