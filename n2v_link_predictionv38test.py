import argparse
import random
import time
import psutil
import sys
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
import pandas as pd
from itertools import combinations

# Argument parser setup
def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate embeddings using logistic regression with edge features.")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the embeddings file.")
    parser.add_argument("--neg_samples", type=str, help="Path to the negative samples file.")
    parser.add_argument("--output", type=str, required=True, help="Output file to save results.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for computations.")
    return parser.parse_args()

# Memory usage check
def check_memory_usage(threshold=0.9):
    mem = psutil.virtual_memory()
    if (mem.percent / 100) >= threshold:
        print(f"Memory usage at {mem.percent}%. Exiting to prevent crashes.")
        sys.exit(1)

# Load embeddings from file (optimized)
def load_embeddings(file_path, device):
    check_memory_usage()
    print(f"Loading embeddings from {file_path}...")
    df = pd.read_csv(file_path, sep=" ", skiprows=1, header=None)
    node_ids = df[0].astype(str)  # Node IDs
    vectors = torch.tensor(df.iloc[:, 1:].values, device=device)
    embeddings = {node_id: vector for node_id, vector in zip(node_ids, vectors)}
    print(f"Loaded {len(embeddings)} embeddings.")
    return embeddings

# Load negative samples from a file
def load_negative_samples(file_path, embeddings):
    print(f"Loading negative samples from {file_path}...")
    negative_samples = []
    with open(file_path, 'r') as f:
        for line in f:
            u, v = line.strip().split()
            if u in embeddings and v in embeddings:
                negative_samples.append((u, v))
    print(f"Loaded {len(negative_samples)} valid negative samples.")
    return negative_samples

def generate_synthetic_negative_samples(nodes, num_samples):
    """
    Generate random negative samples (non-existent edges) for testing.
    """
    print("Generating synthetic negative samples...")
    negative_samples = []
    while len(negative_samples) < num_samples:
        u, v = random.sample(nodes, 2)
        if u != v:  # Avoid self-loops
            negative_samples.append((u, v))
    print(f"Generated {len(negative_samples)} synthetic negative samples.")
    return negative_samples

# Compute edge features (optimized with batch processing)
def compute_edge_features_batch(embeddings, pairs, batch_size=500):
    features = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        u_embs = torch.stack([embeddings[u] for u, _ in batch])
        v_embs = torch.stack([embeddings[v] for _, v in batch])
        
        avg = (u_embs + v_embs) / 2
        hadamard = u_embs * v_embs
        l1 = torch.abs(u_embs - v_embs)
        l2 = torch.square(u_embs - v_embs)
        
        combined = torch.cat([avg, hadamard, l1, l2], dim=1)
        features.append(combined.cpu().numpy())
    
    return np.vstack(features)

# Split data into training and testing (90/10 split) and avoid self-loops
def split_data(embeddings, negative_samples):
    nodes = list(embeddings.keys())

    # Generate positive edges
    total_positive_samples = len(negative_samples)
    positive_edges = set()
    while len(positive_edges) < total_positive_samples:
        u, v = random.sample(nodes, 2)
        if u != v:
            positive_edges.add((u, v))
    positive_edges = list(positive_edges)

    # Ensure a balanced split
    split_idx_pos = int(len(positive_edges) * 0.9)
    split_idx_neg = int(len(negative_samples) * 0.9)

    train_edges = positive_edges[:split_idx_pos] + negative_samples[:split_idx_neg]
    test_edges = positive_edges[split_idx_pos:] + negative_samples[split_idx_neg:]

    train_labels = [1] * split_idx_pos + [0] * split_idx_neg
    test_labels = [1] * (len(positive_edges) - split_idx_pos) + [0] * (len(negative_samples) - split_idx_neg)

    print(f"Split data into {len(train_edges)} training edges and {len(test_edges)} testing edges.")
    return train_edges, train_labels, test_edges, test_labels

# Compute reconstruction loss with batch processing
def compute_reconstruction_loss(embeddings, edges, labels):
    u_embs = torch.stack([embeddings[u] for u, _ in edges])
    v_embs = torch.stack([embeddings[v] for _, v in edges])
    sims = torch.cosine_similarity(u_embs, v_embs, dim=1)
    mse_loss = torch.mean((torch.tensor(labels, device=sims.device) - sims) ** 2)
    return mse_loss.item()

# Train logistic regression and evaluate
def train_and_evaluate(train_features, train_labels, test_features, test_labels):
    print("Training logistic regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(train_features, train_labels)

    y_pred = model.predict_proba(test_features)[:, 1]
    auc_score = roc_auc_score(test_labels, y_pred)
    ap = average_precision_score(test_labels, y_pred)

    return auc_score, ap

# Main function
def main():
    args = parse_arguments()
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading embeddings...")
    embeddings = load_embeddings(args.embeddings, device)

    print("Loading negative samples...")
    if args.neg_samples:
        negative_samples = load_negative_samples(args.neg_samples, embeddings=embeddings)
    else:
        nodes = list(embeddings.keys())
        negative_samples = generate_synthetic_negative_samples(nodes, num_samples=len(nodes) // 2)  # Example size

    print("Splitting data into training and testing...")
    train_edges, train_labels, test_edges, test_labels = split_data(embeddings, negative_samples)

    print("Computing edge features...")
    train_features = compute_edge_features_batch(embeddings, train_edges)
    test_features = compute_edge_features_batch(embeddings, test_edges)

    print("Computing reconstruction loss...")
    reconstruction_loss = compute_reconstruction_loss(embeddings, test_edges, test_labels)

    print("Training and evaluating...")
    start_time = time.time()
    auc_score, ap = train_and_evaluate(train_features, train_labels, test_features, test_labels)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Save results to file
    with open(args.output, "w") as f:
        f.write("Evaluation Metrics\n")
        f.write("==================\n")
        f.write(f"ROC-AUC Score: {auc_score:.4f}\n")
        f.write(f"Average Precision (AP): {ap:.4f}\n")
        f.write(f"Reconstruction Loss (MSE): {reconstruction_loss:.4f}\n")
        f.write(f"Execution Time: {elapsed_time:.2f} seconds\n")

    print("\nResults:")
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")
    print(f"Reconstruction Loss (MSE): {reconstruction_loss:.4f}")
    print(f"Execution Time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
