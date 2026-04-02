import numpy as np
import os
import gc

def compress_mask_file():
    cache_dir = "/data/restanislao/diplomacy/"
    mask_path = os.path.join(cache_dir, "mask.bin")
    targets_path = os.path.join(cache_dir, "targets.bin")
    packed_mask_path = os.path.join(cache_dir, "mask_packed.bin")
    
    num_provs = 82
    
    targets_bytes = os.path.getsize(targets_path)
    total_samples = targets_bytes // (8 * num_provs)
    
    mask_bytes = os.path.getsize(mask_path)
    vocab_size = mask_bytes // (total_samples * num_provs)
    
    print(f"Detected {total_samples} samples and a vocab size of {vocab_size}")
    print("Loading original mask for compression...")
    
    mask_mmap = np.memmap(mask_path, dtype=np.bool_, mode='r', shape=(total_samples, num_provs, vocab_size))
    
    chunk_size = 10000
    total_chunks = (total_samples + chunk_size - 1) // chunk_size
    
    print(f"Packing into {packed_mask_path}...")
    with open(packed_mask_path, 'wb') as f:
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            
            chunk = np.array(mask_mmap[start_idx:end_idx], copy=True)
            chunk_flat = chunk.reshape(end_idx - start_idx, -1)
            
            packed_chunk = np.packbits(chunk_flat, axis=1)
            f.write(packed_chunk.tobytes())
            
            f.flush()
            os.fsync(f.fileno())
            
            del chunk
            del chunk_flat
            del packed_chunk
            gc.collect()
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{total_chunks} chunks")
                
    print("\nCompression complete.")
    old_size = os.path.getsize(mask_path) / (1024**3)
    new_size = os.path.getsize(packed_mask_path) / (1024**3)
    print(f"Reduced mask from {old_size:.2f} GB to {new_size:.2f} GB")

if __name__ == "__main__":
    compress_mask_file()