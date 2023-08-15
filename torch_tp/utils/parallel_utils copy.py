from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, CPUOffload, MixedPrecision, BackwardPrefetch
# from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
# from fairscale.nn.checkpoint import checkpoint_wrapper
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl
import torch.nn as nn
import torch
from .group_comm_utils import gen_allgather_group, gen_slice_func
from typing import Tuple, List
from functools import partial
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../site-package')
from megatron import get_args

# def wrap_data_parallel(module, dp_type = None, dp_group = None, gradient_as_bucket_view = True, broadcast_buffers = True, module_type='bert_enc', pp_device = None, mixed_precision=torch.float32):
#     if dp_type is None:
#         return module
#     elif dp_type == 0:
#         comm_group = None if dp_group is None else dp_group.group
#         return DDP(module, process_group = comm_group, gradient_as_bucket_view=gradient_as_bucket_view, broadcast_buffers=broadcast_buffers)
#     elif dp_type == 1:
#         assert pp_device is not None
#         return wrap_module_fsdp_manually(module, pp_device, module_type, dp_group)
#     else:
#         raise ValueError

def wrap_data_parallel(module, dp_type = None, dp_group = None, module_type='bert_enc', pp_device = None, mixed_precision=torch.bfloat16, pp_on=False):
    if dp_type is None:
        return module
    else:
        assert pp_device is not None
        fsdp_type_dict = {0:get_args().default_dp_type, 1:'zero3'}
        assert dp_type in fsdp_type_dict.keys()
        return wrap_module_fsdp_manually(module, pp_device, module_type, dp_group, fsdp_type=fsdp_type_dict[dp_type], mixed_precision=mixed_precision, pp_on=pp_on)

def wrap_module_fsdp_manually(module, pp_device, module_type='bert_enc', dp_group=None, fsdp_type='zero3', mixed_precision=torch.bfloat16, pp_on=False):
    comm_group = None if dp_group is None else dp_group.group
    sharding_strategy = {'ddp': ShardingStrategy.NO_SHARD,
                           'zero2': ShardingStrategy.SHARD_GRAD_OP,
                           'zero3': ShardingStrategy.FULL_SHARD}[fsdp_type]
    mixed_precision_policy = MixedPrecision(
        param_dtype=mixed_precision, # Param precision
        reduce_dtype=mixed_precision, # Gradient communication precision
        buffer_dtype=mixed_precision, # Buffer precision
        cast_forward_inputs=True,
        cast_root_forward_inputs=True
    )
    backward_prefetch = None if pp_on else BackwardPrefetch.BACKWARD_PRE
    fsdp_args = dict(process_group = comm_group, 
                    sharding_strategy = sharding_strategy, 
                    mixed_precision=mixed_precision_policy, 
                    backward_prefetch=backward_prefetch)

    # if fsdp_type == 'ddp':
    #     return FSDP(module, **fsdp_args)

    if module_type in ['bert_enc', 'vit_enc']:
        sub_module = module.module.layer[0]
        setattr(sub_module, 'attention', FSDP(sub_module.attention.cuda(pp_device), **fsdp_args))
        setattr(sub_module, 'mlp', FSDP(sub_module.mlp.cuda(pp_device), **fsdp_args))
        return FSDP(module, **fsdp_args)
    elif module_type in ['swin_enc']:
        sub_module = module.module.block
        setattr(sub_module, 'attention', FSDP(sub_module.attention.cuda(pp_device), **fsdp_args))
        setattr(sub_module, 'intermediate', FSDP(sub_module.intermediate.cuda(pp_device), **fsdp_args))
        return FSDP(module, **fsdp_args)
    elif module_type in ['t5_enc']:
        sub_module = module.module.block.t5_block
        setattr(sub_module.layer[0], 'SelfAttention', FSDP(sub_module.layer[0].SelfAttention.cuda(pp_device), **fsdp_args))
        sub_module.layer[-1] = FSDP(sub_module.layer[-1].cuda(pp_device), **fsdp_args)
        return FSDP(module, **fsdp_args)
    elif module_type in ['t5_dec']:
        module_ = module.module
        sub_module = module.module.block.t5_block
        setattr(module_, 'block', FSDP(module_.block.cuda(pp_device), **fsdp_args))
        setattr(sub_module.layer[0], 'SelfAttention', FSDP(sub_module.layer[0].SelfAttention.cuda(pp_device), **fsdp_args))
        setattr(sub_module.layer[1], 'EncDecAttention', FSDP(sub_module.layer[1].EncDecAttention.cuda(pp_device), **fsdp_args))
        sub_module.layer[-1] = FSDP(sub_module.layer[-1].cuda(pp_device), **fsdp_args)
        return FSDP(module, **fsdp_args)
    elif module_type in ['gpt_dec']:
        # sub_module = module.module.layers[0]
        # attrs = ['mixer', 'mlp']
        # for key in attrs:
        #     setattr(sub_module, key, FSDP(getattr(sub_module,key).cuda(pp_device), **fsdp_args))
        module.module.layers[0] = FSDP(module.module.layers[0], **fsdp_args)
        return FSDP(module, **fsdp_args)
    else:
        # sub_module =  module.module
        # for name, child in sub_module.named_children():
        #     setattr(sub_module, name, FSDP(child, **fsdp_args))
        return FSDP(module, **fsdp_args)

def wrap_modules_checkpoint(module_list, checkpoint_flags):
    m = module_list
    if isinstance(m, FSDP):
        m = m._fsdp_wrapped_module
    assert len(m) == len(checkpoint_flags)
    for i in range(len(m)):
        if checkpoint_flags[i]:
            m[i] = checkpoint_wrapper(m[i])
            # m[i] = checkpoint_wrapper(m[i], checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    return module_list

def relocate_activations(input, allgather_group, slice_func):
    if allgather_group is None and slice_func is None:
        return input
    if slice_func is not None:
        input = slice_func(input)
    if allgather_group is not None:
        input = allgather_group.allgather(input.contiguous())
    return input

class Module_with_relocation(nn.Module):
    def __init__(self, module, allgather_group, slice_func):
        super().__init__()
        self.module = module
        self.allgather_group = allgather_group
        self.slice_func = slice_func
        self.relocate_activations = lambda x: relocate_activations(x, self.allgather_group, self.slice_func)
        if hasattr(module, 'get_extended_attention_mask'):
            self.get_extended_attention_mask = module.get_extended_attention_mask

    def forward(self, *inputs):
        if isinstance(inputs, (Tuple, List)):
            inputs_relocated = []
            for input in inputs:
                inputs_relocated.append(self.relocate_activations(input))
            inputs_relocated = tuple(inputs_relocated)
            return self.module(*inputs_relocated)
        else:
            input_relocated = self.relocate_activations(inputs)
            return self.module(input_relocated)

def auto_wrap_named_module(module, dp_type, dp_group, name):
    for module_name, child in module.named_children():
        if name in module_name:
            if 'embed' in name:
                wrapped_child = wrap_data_parallel(child, dp_type, dp_group, gradient_as_bucket_view = False, broadcast_buffers = False)
            else:
                wrapped_child = wrap_data_parallel(child, dp_type, dp_group)
            setattr(module, name, wrapped_child)
        else:
            auto_wrap_named_module(child, dp_type, dp_group, name)

def my_auto_wrap(module, dp_type, dp_group):
    module_names = ['embed', 'mlp', 'attention', 'pooler', 'cls', 'layernorm']
    for name in module_names:
        auto_wrap_named_module(module, dp_type, dp_group, name)
    return module

def wrap_modules_data_parallel(module_list, dp_types, dp_groups, module_types, pp_devices=None, mixed_precision=torch.bfloat16, default_process_group=None):
    assert len(module_list) == len(dp_types)
    assert len(module_list) == len(dp_groups)
    
    process_group = default_process_group if default_process_group is not None else dp_groups[0]
    # print(process_group.ranks)
    pp_on = True if process_group.size < torch.distributed.get_world_size() else False
    
    if pp_devices is not None:
        assert len(pp_devices) == len(module_list)
    for i in range(len(module_list)):
        pp_device = None if pp_devices is None else pp_devices[i]
        # # Manual Wrap
        # if 'embed' in module_types[i]:
        #     module_list[i] = wrap_data_parallel(module_list[i], dp_types[i], dp_groups[i], gradient_as_bucket_view = False,  
        #             broadcast_buffers = False, module_type=module_types[i], pp_device = pp_device, mixed_precision=mixed_precision)
        # else:
        #     module_list[i] = wrap_data_parallel(module_list[i], dp_types[i], dp_groups[i], module_type=module_types[i], pp_device = pp_device, mixed_precision=mixed_precision)
        module_list[i] = wrap_data_parallel(module_list[i], dp_types[i], dp_groups[i], module_type=module_types[i], pp_device = pp_device, mixed_precision=mixed_precision, pp_on=pp_on)

    sharding_strategy = {'ddp': ShardingStrategy.NO_SHARD,
                           'zero2': ShardingStrategy.SHARD_GRAD_OP,
                           'zero3': ShardingStrategy.FULL_SHARD}[get_args().default_dp_type]
    # sharding_strategy = ShardingStrategy.NO_SHARD
    mixed_precision_policy = MixedPrecision(
        param_dtype=mixed_precision, # Param precision
        reduce_dtype=mixed_precision, # Gradient communication precision
        buffer_dtype=mixed_precision, # Buffer precision
        cast_forward_inputs=True,
        cast_root_forward_inputs=True
    )
    backward_prefetch = None if pp_on else BackwardPrefetch.BACKWARD_PRE
    fsdp_args = dict(process_group=process_group.group, #torch.distributed.new_group([torch.distributed.get_rank()])
                    sharding_strategy=sharding_strategy, 
                    mixed_precision=mixed_precision_policy, 
                    backward_prefetch=backward_prefetch)
    # if torch.distributed.get_rank() == 0:
    #     print('root fsdp args:', fsdp_args)
    module_list = FSDP(module_list, **fsdp_args)
    # if torch.distributed.get_rank() == 0:
    #     print(module_list)
    return module_list

def modules_to_devices(module_list, pp_devices):
    assert len(module_list) == len(pp_devices)
    for i in range(len(module_list)):
        module_list[i].to('cuda:%d'%pp_devices[i])

def wrap_modules_relocation(module_list, allgather_groups, slice_funcs):
    assert len(module_list) == len(allgather_groups)
    assert len(module_list) == len(slice_funcs)
    for i in range(len(module_list)):
        module_list[i] = Module_with_relocation(module_list[i], allgather_groups[i], slice_funcs[i])
    return module_list

def gen_label_relocation_func(input_tp_size, output_tp_size):
    allgather_group = gen_allgather_group(input_tp_size, output_tp_size, to_print = False)
    slice_func = gen_slice_func(input_tp_size, output_tp_size)
    return lambda label: relocate_activations(label, allgather_group, slice_func)
