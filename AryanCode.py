import torch
from torch_geometric.utils import to_networkx, from_networkx, negative_sampling, subgraph
from torch_geometric.nn.models import GAE
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import networkx as nx

# Load the GML file with NetworkX
nx_graph = nx.read_gml(r"/ceph/home/student.aau.dk/ca48dd/paper2paper_2000_gcc.gml")
print(f"Loaded NetworkX graph with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges.")

# Convert to PyTorch Geometric Data
pyg_data = from_networkx(nx_graph)

# Explicitly set num_nodes
pyg_data.num_nodes = nx_graph.number_of_nodes()

# Initialize node features and labels if they are not present
if not hasattr(pyg_data, 'x') or pyg_data.x is None:
    pyg_data.x = None  # Replace with your feature matrix if available
if not hasattr(pyg_data, 'y') or pyg_data.y is None:
    pyg_data.y = None  # Replace with your labels if available

print(f"Converted to PyTorch Geometric Data with {pyg_data.num_nodes} nodes and {pyg_data.num_edges} edges.")

# Step 1: Load the graph data
graph_data = pyg_data

# Step 2: Validate and preprocess the graph
connected_nodes = torch.unique(graph_data.edge_index)
graph_data.num_nodes = connected_nodes.size(0)

graph_data.edge_index, _ = subgraph(
    connected_nodes, graph_data.edge_index, relabel_nodes=True
)

# Debugging prints before filtering invalid edges
print(f"Before filtering - Max node ID: {graph_data.edge_index.max().item()}")
print(f"Before filtering - Num nodes: {graph_data.num_nodes}")
print(f"Before filtering - Edge Index Shape: {graph_data.edge_index.shape}")

# Filter invalid edges (if any) to ensure they fall within the expected node range
max_expected_nodes = graph_data.num_nodes
valid_edges_mask = (
    (graph_data.edge_index[0] < max_expected_nodes) &
    (graph_data.edge_index[1] < max_expected_nodes)
)
graph_data.edge_index = graph_data.edge_index[:, valid_edges_mask]

# Debugging prints after validation
print(f"After validation - Max node ID: {graph_data.edge_index.max().item()}")
print(f"After validation - Num nodes: {graph_data.num_nodes}")
print(f"After validation - Edge Index Shape: {graph_data.edge_index.shape}")

# Step 3: Assign constant features
graph_data.x = torch.ones((graph_data.num_nodes, 1))  # Single constant feature per node
print(f"Assigned constant features: {graph_data.x.shape}")

# Step 4: Define the GCN encoder
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

# Step 5: Initialize the GCN encoder and GAE model
in_channels = graph_data.x.size(1)
out_channels = 32
encoder = GCN(in_channels, out_channels)
model = GAE(encoder)

# Step 6: Move the model and graph data to the appropriate device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
graph_data = graph_data.to(device)

# Step 7: Encode the graph
z = model.encode(graph_data.x, graph_data.edge_index)
print(f"Encoded graph embeddings shape: {z.shape}")

# Step 8: Generate negative samples for training
pos_edge_index = graph_data.edge_index
neg_edge_index = negative_sampling(
    edge_index=pos_edge_index,
    num_nodes=z.size(0),
    num_neg_samples=pos_edge_index.size(1)
)

# Step 9: Compute the loss for training
loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
print(f"Reconstruction loss: {loss.item()}")

# Step 10: Validation with AUC-PR and AP using PyTorch
with torch.no_grad():
    # Compute predictions for positive and negative edges
    pos_pred = model.decode(z, pos_edge_index)
    neg_pred = model.decode(z, neg_edge_index)
    
    # Create labels for AUC-PR calculation
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
