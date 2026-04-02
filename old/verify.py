import os
import numpy as np

def verify_binary_files(cache_dir="/data/restanislao/diplomacy/"):
    history_path = os.path.join(cache_dir, "history.bin")
    mask_path = os.path.join(cache_dir, "mask.bin")
    targets_path = os.path.join(cache_dir, "targets.bin")
    
    if not (os.path.exists(history_path) and os.path.exists(mask_path) and os.path.exists(targets_path)):
        print("Error One or more binary files are missing.")
        return

    num_provs = 81
    
    targets_bytes = os.path.getsize(targets_path)
    total_samples = targets_bytes // (8 * num_provs)
    
    mask_bytes = os.path.getsize(mask_path)
    vocab_size = mask_bytes // (total_samples * num_provs)
    
    print(f"Detected {total_samples} total samples and a vocabulary size of {vocab_size}.")
    
    try:
        history = np.memmap(history_path, dtype=np.float32, mode='r', shape=(total_samples, 3, num_provs, 16))
        mask = np.memmap(mask_path, dtype=np.bool_, mode='r', shape=(total_samples, num_provs, vocab_size))
        targets = np.memmap(targets_path, dtype=np.int64, mode='r', shape=(total_samples, num_provs))
        print("Successfully loaded all memory maps.")
    except Exception as e:
        print(f"Failed to load memory maps {e}")
        return

    print("\nRunning data sanity checks on the first 1000 samples...")
    sample_limit = min(1000, total_samples)
    
    hist_sample = history[:sample_limit]
    mask_sample = mask[:sample_limit]
    targ_sample = targets[:sample_limit]
    
    if np.isnan(hist_sample).any() or np.isinf(hist_sample).any():
        print("Warning History array contains NaNs or Infs.")
    else:
        print("History array looks clean.")
        
    if targ_sample.max() >= vocab_size or targ_sample.min() < 0:
        print("Warning Target arrays contain out-of-bounds indices.")
    else:
        print("Target arrays are within valid vocabulary bounds.")
        
    if mask_sample.min() < 0 or mask_sample.max() > 1:
        print("Warning Mask array contains non-boolean values.")
    else:
        print("Mask array contains valid boolean data.")

    print("\nVerification complete.")

if __name__ == "__main__":
    verify_binary_files()
