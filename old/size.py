import os

cache_dir = "/data/restanislao/diplomacy/"
history_path = os.path.join(cache_dir, "history.bin")
mask_path = os.path.join(cache_dir, "mask.bin")
targets_path = os.path.join(cache_dir, "targets.bin")

history_bytes = os.path.getsize(history_path)
mask_bytes = os.path.getsize(mask_path)
targets_bytes = os.path.getsize(targets_path)

total_bytes = history_bytes + mask_bytes + targets_bytes
total_gb = total_bytes / (1024**3)

print(f"History RAM needed {history_bytes / (1024**3):.2f} GB")
print(f"Mask RAM needed    {mask_bytes / (1024**3):.2f} GB")
print(f"Targets RAM needed {targets_bytes / (1024**3):.2f} GB")
print(f"Total RAM required {total_gb:.2f} GB")