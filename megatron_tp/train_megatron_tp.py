import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import argparse
from tqdm import tqdm
import numpy as np
import random
import h5py
import time
import os
from utils import print_peak_memory
from megatron.initialize import initialize_megatron
from megatron import get_args, _print_args
from megatron_layers import ParallelMLP, ParallelAttention
from megatron.model.utils import init_method_normal, scaled_init_method_normal
from torch.distributed.distributed_c10d import _get_default_group
from megatron.mpu import get_data_parallel_group, get_tensor_model_parallel_group

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def overwrite_megatron_args(config, args):
    args.hidden_size = config.hidden_size
    args.num_layers = config.num_hidden_layers
    args.num_attention_heads = config.num_attention_heads
    args.ffn_hidden_size = config.intermediate_size
    args.max_position_embeddings = config.max_position_embeddings
    args.attention_dropout = config.attention_probs_dropout_prob
    args.hidden_dropout = config.hidden_dropout_prob
    _print_args(args)

class mlp_tp(nn.Module):
    def __init__(self):
        super().__init__()
        args=get_args()
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.mlp = ParallelMLP(init_method, scaled_init_method)

    def forward(self, hidden_states):
        input_tensor = hidden_states
        hidden_states, bias = self.mlp(hidden_states)
        return hidden_states+bias

def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model =  mlp_tp().to(device)

    # # Obtain dp group and implement data parallel
    # dp_group = get_data_parallel_group()
    # model = torch.nn.parallel.DistributedDataParallel(model, process_group = dp_group)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    if args.profile and rank == 0:
        print_peak_memory("After creating model", rank, args.profile_type)

    if rank == 0:
        print("Start training...")
    for ep in range(10):
            start_time = time.time()
            input = torch.rand(32, 1024).to(device) 
            if args.profile and rank == 0:
                torch.cuda.reset_peak_memory_stats(rank)
                print_peak_memory("\nBefore Forward", rank, args.profile_type)

            out = model(input)
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



def add_arguments(parser):
    group = parser.add_argument_group(title='our arguments')

    group.add_argument(
        "--train_batch_size", type=int, default=32, help="Training batch size"
    )
    group.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    group.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    group.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    group.add_argument(
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    group.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    group.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    group.add_argument("--max_predictions_per_seq", type=int, default=20)
    group.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    group.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )

    group.add_argument(
        "--check_loss", type=int, default=0, help="Whether to check model correctness."
    )
    group.add_argument(
        "--profile", type=int, default=0, help="Whether to profile model GPU memory."
    )
    group.add_argument(
        "--profile_type", type=str, default='allocated', help="Profile allocated memory or reserved memory.",
        choices = ['allocated', 'reserved'],
    )
    parser.add_argument(
        "--load_params", type=int, default=0, help="Whether to load saved init params."
    )

    return parser

if __name__ == '__main__':
    args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'}
    initialize_megatron(extra_args_provider=add_arguments, args_defaults=args_defaults)
    args = get_args()
    set_seed()
    train(args)
