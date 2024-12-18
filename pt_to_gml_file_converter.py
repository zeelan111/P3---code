import torch
import networkx as nx
from torch_geometric.data import Data

def pyg_data_to_gml(pt_file, gml_file):
    """
    Convert a PyTorch Geometric Data object to a .gml file.

    Parameters:
        pt_file (str): Path to the .pt file containing the PyG Data object.
        gml_file (str): Path to save the output .gml file.
    """
    # Load the PyG Data object from the .pt file
    data = torch.load(pt_file)
    
    # Ensure the file contains a PyG Data object
    if not isinstance(data, Data):
        raise TypeError(f"Expected a torch_geometric.data.Data object, but got {type(data)}")
    
    # Initialize a NetworkX graph
    G = nx.Graph()
    
    # Add edges from edge_index
    edge_index = data.edge_index.t().tolist()  # Convert edge tensor to list of tuples
    G.add_edges_from(edge_index)
    
    # Add nodes and their features (if available)
    num_nodes = data.num_nodes if data.num_nodes is not None else data.x.size(0)
    for node_id in range(num_nodes):
        node_attrs = {}
        if data.x is not None:  # Add node features if available
            node_attrs = {f"feature_{i}": data.x[node_id, i].item() for i in range(data.x.size(1))}
        G.add_node(node_id, **node_attrs)
    
    # Save the graph to a .gml file
    nx.write_gml(G, gml_file)
    print(f"Graph successfully saved to {gml_file}")

# Example usage
pyg_data_to_gml('./reduced_graph200.pt', 'reduced_graph200.gml') #input.pt, output.gml
