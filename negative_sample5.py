import argparse
import random
import networkx as nx

def generate_negative_samples_by_id(graph_path, output_path, sample_ratio=1.0):
    """
    Generate negative samples (non-edges) from the graph and save them using the node IDs.

    Parameters:
    - graph_path: Path to the GML graph file.
    - output_path: Path to save the negative samples as ID pairs.
    - sample_ratio: Fraction of edges to sample as negative examples.
    """
    print(f"Loading graph from {graph_path}...")
    # Force NetworkX to use 'id' as the node key instead of 'label'
    graph = nx.read_gml(graph_path, label='id')  # This ensures 'id' is used as the node identifier.

    total_edges = graph.number_of_edges()
    sample_size = int(total_edges * sample_ratio)
    print(f"Sampling {sample_size} negative edges (non-edges)...")

    # Use an iterator to avoid materializing all non-edges
    non_edges = nx.non_edges(graph)

    with open(output_path, "w") as f:
        count = 0
        for u, v in non_edges:
            # Randomly sample non-edges (probabilistic rejection)
            if random.random() < (sample_size / total_edges):
                f.write(f"{u} {v}\n")  # Write the node IDs directly
                count += 1
                if count >= sample_size:
                    break

    print(f"Saved {count} negative samples to {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate negative samples (non-edges) from a graph using node IDs.")
    parser.add_argument("--graph", type=str, required=True, help="Path to the graph (.gml) file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the negative samples.")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Fraction of edges for negative sampling (default: 1.0).")
    args = parser.parse_args()

    generate_negative_samples_by_id(args.graph, args.output, args.sample_ratio)
