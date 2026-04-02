import torch
from diplomacy_helpers import DiplomacyTransformer

def debug_segfault():
    print("Starting minimal single GPU test")
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Check your PyTorch installation.")
        return
        
    device = torch.device("cuda:0")
    print(f"Using device {torch.cuda.get_device_name(0)}")
    
    # Initialize the model on a single GPU without DataParallel
    net = DiplomacyTransformer(num_provinces=81, vocab_size=150).to(device)
    
    # Create a tiny dummy batch
    dummy_history = torch.randn(2, 3, 81, 16, device=device)
    
    print("Running forward pass...")
    try:
        output = net(dummy_history)
        print("Forward pass successful!")
    except Exception as e:
        print(f"Standard Python error caught: {e}")

if __name__ == "__main__":
    debug_segfault()