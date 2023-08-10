import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import argparse
from tqdm import tqdm
import numpy as np
import random
import h5py
import time
import os
import sys
from utils import print_peak_memory
from torch.distributed._tensor.sharding_prop import _CachingPropagator
from torch.distributed.tensor.parallel._utils import _create_1d_device_mesh
from torch.distributed.tensor.parallel.multihead_attention_tp import (
    TensorParallelMultiheadAttention,
)
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    PairwiseParallel,
    ParallelStyle,
    RowwiseParallel,
)
from torch.distributed._tensor import (
    DeviceMesh,
    DTensor,
    distribute_module,
    distribute_tensor,
    Replicate,
    Shard,
)

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(1024, 4096, bias=False)

    def forward(self, hidden_states):
        input_tensor = hidden_states
        hidden_states = self.mlp(hidden_states)
        return hidden_states

def shard_params(mod_name, mod, mesh):
    rowwise_placement = [Shard(0)]
    def to_dist_tensor(t): return distribute_tensor(t, mesh, rowwise_placement)
    mod._apply(to_dist_tensor)

def shard_fc(mod_name, mod, mesh):
    rowwise_placement = [Shard(0)]
    if mod_name == "mlp":
        mod.weight = torch.nn.Parameter(distribute_tensor(mod.weight, mesh, rowwise_placement))

def train(args):
    dist.init_process_group(backend='nccl')
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    world_size = torch.distributed.get_world_size()
    
    rowwise_placement=[Shard(0)]
    mesh = DeviceMesh("cuda", list(range(world_size)))
    sharded_module = distribute_module(mlp(), mesh, partition_fn=shard_fc).to(device)
    
    optimizer = Adam(sharded_module.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    if args.profile and rank == 0:
        print_peak_memory("After creating model", rank, args.profile_type)

    if rank == 0:
        print("Start training...")
    for ep in range(10):
            start_time = time.time()
            input = torch.rand(32, 1024)
            rowwise_tensor = distribute_tensor(input, device_mesh=mesh, placements=rowwise_placement)
            
            if args.profile and rank == 0:
                torch.cuda.reset_peak_memory_stats(rank)
                print_peak_memory("\nBefore Forward", rank, args.profile_type)

            out = sharded_module(rowwise_tensor)
            loss = out.sum()
            if args.profile and rank == 0:
                print_peak_memory("After Forward", rank, args.profile_type)

            loss.backward()

            if args.profile and rank == 0:
                print_peak_memory("After Backward", rank, args.profile_type)
            
            optimizer.step()

            if args.profile and rank == 0:
                print_peak_memory("After optimizer_step", rank, args.profile_type)
            
            optimizer.zero_grad()

            end_time = time.time()
            
            if args.check_loss or args.profile:
                if rank == 0:
                    print('[Epoch %d] Loss = %.3f, Time = %.3f'%(ep, loss.item(), end_time-start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of adam")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--profile", type=int, default=1, help="Whether to profile model GPU memory.")
    parser.add_argument("--profile_type", type=str, default='allocated', help="Profile allocated memory or reserved memory.", choices = ['allocated', 'reserved'])
    parser.add_argument("--check_loss", type=int, default=1, help="Whether to check model correctness.")
    args = parser.parse_args()
    set_seed()
    train(args)
