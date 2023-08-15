import torch
from torch import nn
from torch.optim import Adam
from transformers import BertConfig
import argparse
from tqdm import tqdm
import numpy as np
import random
import h5py
import time
import os
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../site-package')
from utils import print_peak_memory
from megatron.initialize import initialize_megatron
from megatron import get_args
from torch.nn.parallel import DistributedDataParallel as DDP
from megatron_layers import ParallelMLP
from utils import gen_groups_dist, show_groups, wrap_modules_relocation

def init_method_constant(val):
    def init_(tensor):
        return torch.nn.init.constant_(tensor, val)
    return init_

class mlp_tp(nn.Module):
    def __init__(self, tp_group, layer_idx, rank):
        super().__init__()
        args=get_args()
        self.tp_group = tp_group
        self.layer_idx = layer_idx
        init_method = init_method_constant(1)
        scaled_init_method = init_method_constant(1)
        self.mlp = ParallelMLP(init_method, scaled_init_method, tp_group = self.tp_group.group)
        self.rank = rank

    def forward(self, hidden_states):
        # pause to check output on each rank
        # rank = torch.distributed.get_rank()
        # if self.layer_idx == 0:
        #     time.sleep(self.rank*0.01)
        # else:
        #     time.sleep(0.08)
        # print("%d input: rank %d"%(self.layer_idx, self.rank), hidden_states.detach().cpu().numpy().tolist())

        hidden_states, _ = self.mlp(hidden_states)

        # pause to check output on each rank
        # rank = torch.distributed.get_rank()
        # time.sleep(0.08)
        # print("%d output: rank %d"%(self.layer_idx, self.rank), hidden_states.detach().cpu().numpy().tolist())

        return hidden_states

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
    args.add_bias_linear = False
    args.bias_gelu_fusion = False
    args.async_tensor_model_parallel_allreduce = False
    #_print_args(args)

def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    print("Creating Model...")
    config = BertConfig(vocab_size=args.vocab_size, 
                        hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers, 
                        num_attention_heads=args.num_attention_heads, 
                        intermediate_size=args.hidden_size*4, 
                        max_position_embeddings=args.seq_length, 
                        attention_probs_dropout_prob=args.dropout_prob,
                        hidden_dropout_prob=args.dropout_prob)
    overwrite_megatron_args(config, args)

    ### select one config to test
    # # tp_size go up test
    # consec test
    # all_tp_sizes =          [1,2,4,8,4,2,1,2,4,1]
    all_tp_sizes =          [1,1,1,1,1,1,1,1,1,1]
    # all_tp_sizes =          [8,8,8,8,8,8,8,8,8,8]
    tp_consecutive_flags =  [1,1,1,1,1,1,1,1,1,1]
    # # inconsec test
    # all_tp_sizes =          [1,2,4,8,1,4,1,8,2,8]
    # tp_consecutive_flags =  [0,0,0,0,0,0,0,0,0,0]
    # # consec & inconsec test
    # all_tp_sizes =          [1,2,2,2,4,2,4,4,4,8]
    # tp_consecutive_flags =  [1,1,0,1,0,0,1,0,1,0]

    # # tp_size go down test
    # # consec test
    # all_tp_sizes =          [1,8,4,2,1,8,2,8,1,4,1]
    # tp_consecutive_flags =  [1,1,1,1,1,1,1,1,1,1,1]
    # # inconsec test
    # all_tp_sizes =          [1,8,4,2,1,8,2,8,1,4,1]
    # tp_consecutive_flags =  [0,0,0,0,0,0,0,0,0,0,0]
    # # consec & inconsec test
    # all_tp_sizes =          [1,8,4,2,4,2,1]
    # tp_consecutive_flags =  [1,1,1,0,0,1,1]

    # # tp_size stay same test
    # # consec test
    # all_tp_sizes =          [1,1,2,2,4,4,8,8]
    # tp_consecutive_flags =  [1,1,1,1,1,1,1,1]
    # # inconsec test
    # all_tp_sizes =          [1,1,2,2,4,4,8,8]
    # tp_consecutive_flags =  [0,0,0,0,0,0,0,0]
    
    use_dtensor = True
    
    _, tp_groups, dp_groups, allgather_groups, slice_funcs = gen_groups_dist(all_tp_sizes, 1, tp_consecutive_flags, use_dtensor=use_dtensor)

    # print groups by rank
    # time.sleep(rank*0.1)
    # print("rank %d TP groups:"%rank)
    # show_groups(tp_groups)
    # time.sleep(0.8)
    # print("rank %d DP groups:"%rank)
    # show_groups(dp_groups)
    # if not use_dtensor:
    #     time.sleep(0.8)
    #     print("rank %d AllGather groups:"%rank)
    #     show_groups(allgather_groups)
    #     time.sleep(0.8)
    #     print("rank %d SliceFuncs:"%rank)
    #     show_groups(slice_funcs)

    model = nn.Sequential()
    for i in range(len(all_tp_sizes)):
        module = DDP(mlp_tp(tp_group=tp_groups[i], 
                            layer_idx=i, rank=rank).cuda(), 
                    process_group=dp_groups[i].group, gradient_as_bucket_view=True)
        model.add_module('mlp_%d'%i, module)

    model = wrap_modules_relocation(model, allgather_groups, slice_funcs, use_dtensor=use_dtensor)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    if args.profile and rank == 0:
        print_peak_memory("After creating model", rank, args.profile_type)

    print("Start training...")
    for ep in range(2):
        start_time = time.time()
        input = torch.randn(1, 1, 2).to(device) * (rank+1)
        if args.profile and rank == 0:
            torch.cuda.reset_peak_memory_stats(rank)
            print_peak_memory("\nBefore Forward", rank, args.profile_type)

        out = model(input)
        loss = out.sum()
        if args.profile and rank == 0:
            print_peak_memory("After Forward", rank, args.profile_type)

        loss.backward()
        
        # if args.profile and rank == 0:
        #     print_peak_memory("After Backward", rank, args.profile_type)
        
        # Check Grad
        print("Model Grad after backward pass:")
        if rank == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:  # Only print trainable parameters   
                    print()
                    print(param.grad)
        
        # Print model parameters after backward propagation
        # print("Model Parameters after backward pass:")
        # if rank == 0:
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:  # Only print trainable parameters
        #             print(param.data)

        optimizer.step()

        # if args.profile and rank == 0:
        #     print_peak_memory("After optimizer_step", rank, args.profile_type)
        
        optimizer.zero_grad()

        # end_time = time.time()
        # if args.check_loss or args.profile:
        #     print('[Epoch %d] Loss = %.3f, Time = %.3f'%(ep, loss.item(), end_time-start_time))


def add_arguments(parser):
    group = parser.add_argument_group(title='our arguments')

    parser.add_argument(
        "--local-rank", type=int, default=-1, help="Local rank.",
    )
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
    initialize_megatron(extra_args_provider=add_arguments)
    args = get_args()
    set_seed()
    train(args)
