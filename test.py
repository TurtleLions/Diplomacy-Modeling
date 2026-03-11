import torch
import torch.nn as nn
import gc
from diplomacy_helpers import DiplomacyTransformer

def find_max_batch_size(num_provs=81, vocab_size=150, starting_batch=32, step=32):
    device = torch.device("cuda:0")
    print(f"Testing limits on a single GPU {torch.cuda.get_device_name(0)}")
    
    net = DiplomacyTransformer(num_provinces=num_provs, vocab_size=vocab_size).to(device)
    net.train()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    current_batch = starting_batch
    max_successful_batch = 0
    
    while True:
        try:
            print(f"Testing batch size {current_batch}...")
            
            dummy_history = torch.randn(current_batch, 3, num_provs, 16, device=device)
            dummy_targets = torch.randint(0, vocab_size, (current_batch, num_provs), device=device)
            
            optimizer.zero_grad()
            logits, _ = net(dummy_history)
            
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = dummy_targets.view(-1)
            
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()
            
            max_successful_batch = current_batch
            current_batch += step
            
            del dummy_history, dummy_targets, logits, logits_flat, loss
            torch.cuda.empty_cache()
            gc.collect()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Caught Out of Memory at batch size {current_batch}.")
                torch.cuda.empty_cache()
                gc.collect()
                break
            else:
                raise e
                
    # Calculate safe capacity for one GPU
    safe_single_gpu_batch = int((max_successful_batch) * 0.9)
    safe_single_gpu_batch = safe_single_gpu_batch - (safe_single_gpu_batch % 8)
    
    # Scale up for your full server
    total_gpus = torch.cuda.device_count()
    final_global_batch = safe_single_gpu_batch * total_gpus
    
    print("\nDiagnostic complete")
    print(f"Max capacity per GPU is {max_successful_batch}")
    print(f"Recommended global batch size for {total_gpus} GPUs is {final_global_batch}")
    
    del net, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    find_max_batch_size(num_provs=81, vocab_size=150)