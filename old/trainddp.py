import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# --- MODEL ARCHITECTURE ---

class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.proj = nn.Linear(in_f, out_f)
    def forward(self, x, adj):
        # Optimized for bfloat16 matrix multiplication
        return torch.relu(torch.matmul(adj, self.proj(x)))

class DiplomacyGCN(nn.Module):
    def __init__(self, adj, in_dim=16, hid_dim=256, vocab_size=82):
        super().__init__()
        self.register_buffer('adj', adj)
        self.gcn1 = GCNLayer(in_dim, hid_dim)
        self.gcn2 = GCNLayer(hid_dim, hid_dim)
        self.type_head = nn.Linear(hid_dim, 8)
        self.t1_head = nn.Linear(hid_dim, vocab_size)
        self.t2_head = nn.Linear(hid_dim, vocab_size)

    def forward(self, x):
        h = self.gcn1(x, self.adj)
        h = self.gcn2(h, self.adj)
        return self.type_head(h), self.t1_head(h), self.t2_head(h)

# --- DATA & UTILS ---

class DiplomacyDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def get_adj():
    from diplomacy import Game
    import numpy as np
    g = Game(); provinces = g.map.locs
    p_idx = {p: i for i, p in enumerate(provinces)}
    adj = np.eye(len(provinces), dtype=np.float32)
    for loc, neighbors in g.map.loc_abut.items():
        u = loc.split('/')[0].upper()
        if u in p_idx:
            for n in neighbors:
                v = n.split('/')[0].upper()
                if v in p_idx: adj[p_idx[u], p_idx[v]] = 1.0
    d_inv = np.power(adj.sum(1), -0.5)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = np.diag(d_inv)
    return torch.from_numpy(d_mat @ adj @ d_mat)

# --- TRAINING LOOP ---

def train():
    dist.init_process_group(backend="nccl") # RCCL uses nccl alias
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    # 1. Load Data (Loads tensors directly to CPU RAM, then moves per batch)
    dataset = DiplomacyDataset("processed_diplomacy.pt")
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=128, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True)

    # 2. Setup Model
    adj = get_adj().to(rank).to(torch.bfloat16)
    model = DiplomacyGCN(adj).to(rank).to(torch.bfloat16)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(10):
        sampler.set_epoch(epoch)
        model.train()
        
        for i, batch in enumerate(loader):
            # Efficiently move data to GPU
            state = batch['state'].to(rank, non_blocking=True).to(torch.bfloat16)
            targets = batch['targets'].to(rank, non_blocking=True)
            m_type = batch['mask_type'].to(rank, non_blocking=True)
            m_t1 = batch['mask_t1'].to(rank, non_blocking=True)
            m_t2 = batch['mask_t2'].to(rank, non_blocking=True)

            optimizer.zero_grad(set_to_none=True) # Slightly faster than zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                t_logits, t1_logits, t2_logits = model(state)
                
                # Flatten for loss
                t_l, t1_l, t2_l = t_logits.view(-1, 8), t1_logits.view(-1, 82), t2_logits.view(-1, 82)
                mt, mt1, mt2 = m_type.view(-1, 8), m_t1.view(-1, 82), m_t2.view(-1, 82)
                targs = targets.view(-1, 3)

                # Identify provinces with units that have legal orders
                valid = (targs[:, 0] != 0)
                
                if valid.any():
                    # Masking logits for invalid engine moves
                    l_type = criterion(t_l.masked_fill(~mt, -1e4)[valid], targs[valid, 0])
                    l_t1 = criterion(t1_l.masked_fill(~mt1, -1e4)[valid], targs[valid, 1])
                    l_t2 = criterion(t2_l.masked_fill(~mt2, -1e4)[valid], targs[valid, 2])
                    loss = l_type + l_t1 + l_t2

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()

            if i % 100 == 0 and rank == 0:
                print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f}")

    if rank == 0:
        torch.save(model.module.state_dict(), "diplomacy_gcn_final.pth")
    dist.destroy_process_group()

if __name__ == "__main__":
    train()