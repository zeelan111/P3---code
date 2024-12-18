import os
import torch
import networkx as nx
from torch_geometric.utils import from_networkx, remove_self_loops, to_undirected
from torch_geometric.data import Data


def load_gml_to_pyg(path_to_gml: str) -> Data:
    """
    Load a GML file using NetworkX and convert it to PyTorch Geometric Data.
    
    Parameters:
    - path_to_gml (str): Path to the GML file.

    Returns:
    - pyg_data (Data): PyTorch Geometric graph data object.
    """
    # Load the GML file with NetworkX
    nx_graph = nx.read_gml(path_to_gml)
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
    return pyg_data


def create_smaller_sample_pyg(data: Data, target_node_count: int, save_path: str) -> Data:
    """
    Create a smaller sample graph using PyTorch Geometric.
    
    Parameters:
    - data (Data): PyTorch Geometric graph data object.
    - target_node_count (int): Target number of nodes in the reduced graph.
    - save_path (str): Path to save the resulting graph.

    Returns:
    - sampled_data (Data): Reduced graph as a PyTorch Geometric data object.
    """
    print(f"Original Graph: {data}")
    print(f"Original Node Count: {data.num_nodes}, Original Edge Count: {data.num_edges}")

    # Remove self-loops for cleaner processing
    data.edge_index, _ = remove_self_loops(data.edge_index)
    print(f"Self-loops removed. Edge Count: {data.num_edges}")

    # Ensure the graph is undirected
    data.edge_index = to_undirected(data.edge_index)
    print("Converted graph to undirected.")

    # Randomly sample nodes until the target count is reached
    remaining_nodes = torch.randperm(data.num_nodes)[:target_node_count]
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[remaining_nodes] = True

    # Mask the edges to only include connections between sampled nodes
    edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
    reduced_edge_index = data.edge_index[:, edge_mask]

    # Create the new reduced data object
    sampled_data = Data(
        x=data.x[mask] if data.x is not None else None,  # Features (if any)
        edge_index=reduced_edge_index,  # Reduced edges
        y=data.y[mask] if data.y is not None else None  # Labels (if any)
    )

    # Explicitly set the number of nodes in the reduced graph
    sampled_data.num_nodes = mask.sum().item()

    print(f"Reduced Graph: {sampled_data}")
    print(f"Reduced Node Count: {sampled_data.num_nodes}, Reduced Edge Count: {sampled_data.num_edges}")

    # Save the graph
    if save_path:
        torch.save(sampled_data, save_path)
        print(f"Graph saved to {save_path}")

    return sampled_data


# Example usage:
if __name__ == "__main__":
    # Path to your GML file
    path_to_gml = r"/ceph/home/student.aau.dk/ca48dd/paper2paper_2000_gcc.gml"

    # Load the GML file and convert to PyTorch Geometric
    pyg_data = load_gml_to_pyg(path_to_gml)

    # Create a smaller graph with target node count
    target_node_count = 1000  # Adjust as needed

    # Use SLURM_TMPDIR or fallback to a temporary directory for saving
    save_path = os.path.join(os.environ.get(r'/ceph/home/student.aau.dk/ca48dd/', r'/ceph/home/student.aau.dk/ca48dd/'), "reduced_graph200.pt")

    # Create a smaller sample
    reduced_data = create_smaller_sample_pyg(pyg_data, target_node_count, save_path)
