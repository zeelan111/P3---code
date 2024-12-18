import torch
from torch_geometric.utils import to_networkx, from_networkx, negative_sampling, subgraph
from torch_geometric.nn.models import GAE
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import networkx as nx


# Step 1: Load the graph data
graph_data = torch.load(r"/ceph/home/student.aau.dk/ca48dd/GCN/reduced_graph800.pt")

# Step 2: Validate and preprocess the graph
# Ensure num_nodes reflects the actual number of unique nodes
connected_nodes = torch.unique(graph_data.edge_index)
graph_data.num_nodes = connected_nodes.size(0)

# Relabel nodes to ensure consecutive IDs
graph_data.edge_index, _ = subgraph(
    connected_nodes, graph_data.edge_index, relabel_nodes=True
)

# Debugging prints before filtering invalid edges
print(f"Before filtering - Max node ID: {graph_data.edge_index.max().item()}")
print(f"Before filtering - Num nodes: {graph_data.num_nodes}")
print(f"Before filtering - Edge Index Shape: {graph_data.edge_index.shape}")

# Filter invalid edges (if any) to ensure they fall within the expected node range
max_expected_nodes = graph_data.num_nodes  # Based on actual connected nodes
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
in_channels = graph_data.x.size(1)  # Input feature size
out_channels = 32  # Size of the output embeddings
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

# Step 10: Optional - Validation or testing (implement based on your needs)
with torch.no_grad():
    val_auc, val_ap = model.test(z, pos_edge_index, neg_edge_index)
    print(f"Validation AUC: {val_auc:.4f}, Validation AP: {val_ap:.4f}")
