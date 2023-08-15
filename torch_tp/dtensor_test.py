import torch
from torch.distributed._tensor import DTensor, DeviceMesh, Shard, Replicate, distribute_tensor, distribute_module
import argparse
from utils import show_groups, wrap_modules_relocation, gen_groups_dist, relocate_activations_dtensor
import time


# This is the script to test the tp change in model forwarding ###

def test(args):
    torch.distributed.init_process_group(backend="nccl")
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    all_tp_sizes =          [1,2,2,4,2] # determine the tp size for each operator: \
            # tp=1 for op1; tp=2 for op2; tp=2 for op3; tp=4 for op4; tp=2 for op5.
    tp_consecutive_flags =  [1,1,1,1,1] # can be ignored
    pp_size = 2
            # specify the pp size to be 2
    num_pp_groups = world_size // pp_size
    pp_stage_id = int(rank // num_pp_groups)
    min_stage_rank, max_stage_rank = pp_stage_id * num_pp_groups, (pp_stage_id + 1) * num_pp_groups
    
    # see details in gen_group_dist api in ./utils
    _, tp_groups, dp_groups, allgather_meshes, slice_meshes = gen_groups_dist(all_tp_sizes, pp_size, tp_consecutive_flags, use_dtensor=True)

    input = torch.ones(1, 2).to(device) * (rank+1)
    
    # specify shard placement
    shard_placement=[Shard(0),Shard(0),Shard(0)]
    # specify replica placement
    replica_placement = [Shard(0),Shard(0),Replicate()]
    inputs = [input]
    outputs = []
    
    def get_device_mesh_list(tp_group_all):
        device_mesh_list = []
        for i in range(pp_size):
            device_mesh_list.append([])
        for g in tp_group_all:
            for i in range(pp_size):
                if i*num_pp_groups <= g.ranks[0] < (i+1)*num_pp_groups:
                    device_mesh_list[i].append(g.ranks)
        return device_mesh_list
    
    for i in range(1, len(all_tp_sizes)):
        device_mesh_old, device_mesh_new = slice_meshes[i], allgather_meshes[i]
        device_mesh_list_old = device_mesh_old.mesh.numpy().tolist() if device_mesh_old is not None else None
        device_mesh_list_new = device_mesh_new.mesh.numpy().tolist() if device_mesh_new is not None else None
        
        # device_mesh_old = DeviceMesh("cuda", device_mesh_list_old)
        # device_mesh_new = DeviceMesh("cuda", device_mesh_list_new)
        
        if rank in [0]:
            print('device_mesh_old:', device_mesh_list_old)
            print('device_mesh_new:', device_mesh_list_new)

        if rank >= num_pp_groups:
            input = inputs[i-1]
            
            # allgather and slice are used for the gather/slice of intermediate activation between different op with different tp size ###
            # see details in relocate_activations_dtensor api in ./utils
            output = relocate_activations_dtensor(input, allgather_meshes[i], slice_meshes[i])
            
            
            
            # if device_mesh_old is not None:
            #     input_dtensor = DTensor.from_local(input, device_mesh_old, replica_placement)
            #     print(input_dtensor)
            #     sharded = input_dtensor.redistribute(device_mesh_old, shard_placement).to_local()
            # else:
            #     sharded = input

            # sharded_dtensor = DTensor.from_local(sharded, device_mesh_new, shard_placement)
            # # print(sharded_dtensor)
            # output_dtensor = sharded_dtensor.redistribute(device_mesh_new, replica_placement)
            # output = output_dtensor.to_local()
            # print(output_dtensor)
            
            
            
            outputs.append(output)
            inputs.append(output)
            
            print(output)

            time.sleep(0.8)
            if rank == 0:
                print('============================')



    # # construct a device mesh with available devices (multi-host or single host)
    # # device_mesh = DeviceMesh("cuda", [0, 1, 2, 3])
    # device_mesh = DeviceMesh("cuda", [[0, 1],[2,3]])
    # # if we want to do row-wise sharding
    # rowwise_placement=[Shard(0),Replicate()]

    # # big_tensor = torch.randn(4, 2)
    
    # big_tensor = torch.ones(4, 2).to(device) * (rank+1)
    
    # # distributed tensor returned will be sharded across the dimension specified in placements
    # rowwise_tensor = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=rowwise_placement)
    # print(rank, rowwise_tensor)
    
    
    
    

    # # if we want to do replication across a certain device list
    # replica_placement = [Replicate(),Replicate()]
    
    
    # replica_tensor = rowwise_tensor.redistribute(device_mesh, replica_placement)
    # print(replica_tensor)
    
    
    # # # distributed tensor will be replicated to all four GPUs.
    # # replica_tensor = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=replica_placement)
    # # print(rank, replica_tensor)

    # # # create a DistributedTensor that shards on dim 0, from a local torch.Tensor
    # # rowwise_tensor = DTensor.from_local(big_tensor, device_mesh, replica_placement)
    # # print(rank, rowwise_tensor)


def add_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local-rank", type=int, default=-1, help="Local rank.",
    )
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = add_arguments()
    test(args)
