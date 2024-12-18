import networkx as nx
from node2vec import Node2Vec
import torch
import argparse
import os
import time
import multiprocessing

# Start the timer
start_time = time.time()

def load_graph(gml_file):
    """Load a graph from a GML file."""
    try:
        if not os.path.exists(gml_file):
            raise FileNotFoundError(f"File {gml_file} not found.")
        if not gml_file.endswith('.gml'):
            raise ValueError("Input file must be in GML format.")
        graph = nx.read_gml(gml_file)
        print(f"Graph loaded successfully with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
        return graph
    except Exception as e:
        print(f"Error loading graph: {e}")
        raise

def generate_embeddings(graph, dimensions=64, walk_length=10, num_walks=10, workers=32, q=1.0, p=1.0):
    """Generate node embeddings using Node2Vec."""
    print("Initializing Node2Vec...")
    try:
        node2vec = Node2Vec(
            graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            p=p,
            q=q
        )

        print("Fitting Node2Vec model...")
        model = node2vec.fit(window=6, min_count=1, batch_words=8)

        embeddings = torch.tensor(model.wv.vectors)
        print(f"Embeddings computation complete. Shape: {embeddings.shape}")

        return model, embeddings
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        raise

def save_embeddings(model, output_file):
    """Save embeddings to a file."""
    try:
        if os.path.exists(output_file):
            overwrite = input(f"{output_file} exists. Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("Operation aborted.")
                return
        print(f"Saving embeddings to {output_file}...")
        model.wv.save_word2vec_format(output_file)
        print("Embeddings saved successfully.")
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Node2Vec on GML file")
    # Positional arguments
    parser.add_argument('gml_file', type=str, help="Path to the input GML file")
    parser.add_argument('output_file', type=str, help="Path to save the embeddings")
    # Optional arguments
    parser.add_argument('--dimensions', type=int, default=64, help="Number of dimensions for embeddings")
    parser.add_argument('--walk_length', type=int, default=10, help="Length of random walks")
    parser.add_argument('--num_walks', type=int, default=10, help="Number of random walks per node")
    parser.add_argument('--workers', type=int, default=32, help="Number of workers for parallel processing")
    parser.add_argument('--qvalue', type=float, default=1.0, help="The value of q in the Node2Vec algorithm")
    parser.add_argument('--pvalue', type=float, default=1.0, help="The value of p in the Node2Vec algorithm")

    args = parser.parse_args()

    # Validate workers count
    max_workers = multiprocessing.cpu_count()
    if args.workers > max_workers:
        print(f"Reducing workers to the maximum available: {max_workers}")
        args.workers = max_workers

    # Validate other input arguments
    if args.dimensions <= 0 or args.walk_length <= 0 or args.num_walks <= 0:
        raise ValueError("Dimensions, walk length, and number of walks must be positive integers.")

    # Load the graph
    graph = load_graph(args.gml_file)

    # Generate embeddings
    model, embeddings = generate_embeddings(
        graph,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        workers=args.workers,
        q=args.qvalue,
        p=args.pvalue
    )

    # Save the embeddings
    save_embeddings(model, args.output_file)

    # Stop the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution Time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()

