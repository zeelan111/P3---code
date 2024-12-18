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

# Step 1: Compute Node Degree
print("Computing node degrees...")
degree_dict = dict(nx_graph.degree())
degree_values = torch.tensor(list(degree_dict.values()), dtype=torch.float)

# Normalize Node Degree
min_degree = torch.min(degree_values)
max_degree = torch.max(degree_values)
normalized_degrees = (degree_values - min_degree) / (max_degree - min_degree)

# Step 2: Compute PageRank
print("Computing PageRank...")
pagerank_scores = nx.pagerank(nx_graph)
pagerank_values = torch.tensor(list(pagerank_scores.values()), dtype=torch.float)

# Normalize PageRank
min_pagerank = torch.min(pagerank_values)
max_pagerank = torch.max(pagerank_values)
normalized_pagerank = (pagerank_values - min_pagerank) / (max_pagerank - min_pagerank)

# Map string node IDs to consecutive integer indices
node_mapping = {node_id: idx for idx, node_id in enumerate(degree_dict.keys())}

# Step 3: Combine Features
print("Combining Node Degree and PageRank...")
combined_features = torch.zeros((pyg_data.num_nodes, 2))  # Two features: degree and PageRank
for node_id, idx in node_mapping.items():
    combined_features[idx, 0] = normalized_degrees[idx]  # Assign normalized degree
    combined_features[idx, 1] = normalized_pagerank[idx]  # Assign normalized PageRank

pyg_data.x = combined_features  # Assign combined features
print(f"Assigned combined features: {pyg_data.x.shape}")

# Step 4: Validate and preprocess the graph
connected_nodes = torch.unique(pyg_data.edge_index)
pyg_data.num_nodes = connected_nodes.size(0)

# Relabel nodes to ensure consecutive IDs
pyg_data.edge_index, _ = subgraph(
    connected_nodes, pyg_data.edge_index, relabel_nodes=True
)

# Debugging prints before filtering invalid edges
print(f"Before filtering - Max node ID: {pyg_data.edge_index.max().item()}")
print(f"Before filtering - Num nodes: {pyg_data.num_nodes}")
print(f"Before filtering - Edge Index Shape: {pyg_data.edge_index.shape}")

# Filter invalid edges
max_expected_nodes = pyg_data.num_nodes  # Based on actual connected nodes
valid_edges_mask = (
    (pyg_data.edge_index[0] < max_expected_nodes) &
    (pyg_data.edge_index[1] < max_expected_nodes)
)
pyg_data.edge_index = pyg_data.edge_index[:, valid_edges_mask]

# Debugging prints after validation
print(f"After validation - Max node ID: {pyg_data.edge_index.max().item()}")
print(f"After validation - Num nodes: {pyg_data.num_nodes}")
print(f"After validation - Edge Index Shape: {pyg_data.edge_index.shape}")

# Step 5: Define the GCN encoder
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


# Step 6: Initialize the GCN encoder and GAE model
in_channels = pyg_data.x.size(1)  # Input feature size
out_channels = 32  # Size of the output embeddings
encoder = GCN(in_channels, out_channels)
model = GAE(encoder)

# Step 7: Move the model and graph data to the appropriate device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
pyg_data = pyg_data.to(device)

# Step 8: Encode the graph
z = model.encode(pyg_data.x, pyg_data.edge_index)
print(f"Encoded graph embeddings shape: {z.shape}")

# Step 9: Generate negative samples for training
pos_edge_index = pyg_data.edge_index
neg_edge_index = negative_sampling(
    edge_index=pos_edge_index,
    num_nodes=z.size(0),
    num_neg_samples=pos_edge_index.size(1)
)

# Step 10: Compute the loss for training
loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
print(f"Reconstruction loss: {loss.item()}")

# Step 11: Optional - Validation or testing
with torch.no_grad():
    val_auc, val_ap = model.test(z, pos_edge_index, neg_edge_index)
    print(f"Validation AUC: {val_auc:.4f}, Validation AP: {val_ap:.4f}")
