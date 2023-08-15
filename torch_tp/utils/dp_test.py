
from dp_utils_dist import DPAlg
import numpy as np
import time

def build_v_and_cost_for_bert(layer_num):

    # index list:
    # 0 -> (1, 1, 8)
    # 1 -> (1, 2, 4)
    # 2 -> (1, 4, 2)
    # 3 -> (1, 8, 1)
    v = np.array([278, 192, 163, 171]).reshape(1, -1).repeat(layer_num, axis=0) # dtype should be np.int32

    # following dtypes could be np.float64
    # calculated purely by dp+tp
    intra_layer_cost = np.array([84, 52, 60, 112], dtype=np.float64).reshape(1, -1).repeat(layer_num, axis=0)

    # row axis: current layer
    # column axis: last layer
    ### NOTE: the correctness of this matrix is uncertained
    inter_layer_cost = np.array([
        [0, 2, 6, 14],
        [0 ,0 ,4 ,12],
        [0, 0, 0, 8], 
        [0, 0, 0, 0]
    ], dtype=np.float64)

    inter_layer_cost = np.expand_dims(inter_layer_cost, axis=0).repeat(layer_num, axis=0)
    inter_layer_cost[0, :, :] = 0 # no inter-layer communication cost in first layer

    inter_layer_cost = inter_layer_cost

    # returned values are supposed to be related to bsz, pp_deg
    return v, intra_layer_cost, inter_layer_cost

if __name__ == '__main__':
    use_cpp_core = True
    layer_num = 24
    mem_lists = [m*1024 for m in [16]]
    for max_mem in mem_lists:
        print("Testing with max_mem=%d"%max_mem)
        dpAlg = DPAlg(max_mem=max_mem, use_cpp_core=use_cpp_core)
        dpAlg.set_v_and_cost(*build_v_and_cost_for_bert(layer_num))
        start_time = time.time()
        comm_cost, res_list, mem_remain = dpAlg.fit()
        end_time = time.time()
        print("Search time is: %.4f s"%(end_time-start_time))
        print("time cost:", comm_cost)
        print("memory remaining:", mem_remain)
        print(res_list)

        print("=======================================")