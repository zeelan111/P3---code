import torch

def inspect_pt_file(file_path):
    """Inspect the structure and contents of a .pt file."""
    try:
        # Load the .pt file
        data = torch.load(file_path)
        
        # Check the type of the data
        print(f"Type of data: {type(data)}")
        
        # Handle dictionary-like data
        if isinstance(data, dict):
            print("Keys in the data:", data.keys())
            for key, value in data.items():
                print(f"Key: {key}, Type: {type(value)}")
                if isinstance(value, list):
                    print(f"  First 5 elements: {value[:5]}")
                elif isinstance(value, torch.Tensor):
                    print(f"  Tensor shape: {value.shape}")
                else:
                    print(f"  Value: {value}")
        
        # Handle other formats
        else:
            print("Data content:", data)
    
    except Exception as e:
        print(f"Error loading the .pt file: {e}")

# Example usage
# inspect_pt_file("./example_graph.pt")
inspect_pt_file("./reduced_graph100.pt")