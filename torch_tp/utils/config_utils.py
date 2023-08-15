import json
import os
from .dp_utils import form_strategy
from typing import List

def str2array(s):
    return list(map(int,s.split(',')))

def array2str(a):
    return ",".join(map(str,a))

def read_json_config(path):
    return json.load(open(path,'r',encoding="utf-8"))

def write_json_config(config, path):
    with open(path,'w') as fp:
        json.dump(config,fp, indent=4)

def config2strategy(config):
    pp_deg = config['pp_deg']
    tp_sizes_enc = str2array(config['tp_sizes_enc'])
    tp_consecutive_flags = str2array(config['tp_consecutive_flags'])
    dp_types_enc = str2array(config['dp_types_enc'])
    return pp_deg, tp_sizes_enc, tp_consecutive_flags, dp_types_enc

def strategy2config(strategy_list):
    layer_num = len(strategy_list)
    if layer_num == 0:
        return {}
    pp_deg = strategy_list[0][0]
    tp_sizes_enc = array2str([s[1] for s in strategy_list])
    tp_consecutive_flags = array2str([0 if 'tp' in s[-1] and not s[-1]['tp'] else 1 for s in strategy_list])
    dp_types_enc = array2str([1 if 'fsdp' in s[-1] and s[-1]['fsdp'] else 0 for s in strategy_list])
    config = {"pp_deg":pp_deg, "tp_sizes_enc":tp_sizes_enc, "tp_consecutive_flags":tp_consecutive_flags, "dp_types_enc":dp_types_enc}
    return config

def read_allreduce_bandwidth_config(config_path, gpu_num):
    env_config = read_json_config(config_path)
    comm_coe_dict={}
    pp_deg = 1
    while pp_deg <= gpu_num:
        comm_coe_dict[pp_deg]={}
        max_dp = gpu_num // pp_deg
        if max_dp >= 2:
            comm_coe_dict[pp_deg]['%d'%max_dp]=env_config['%d_%d_1'%(pp_deg, max_dp)]
        max_dp = max_dp // 2
        while max_dp >= 2:
            comm_coe_dict[pp_deg]['%d_0'%max_dp]=env_config['%d_%d_0'%(pp_deg, max_dp)]
            comm_coe_dict[pp_deg]['%d_1'%max_dp]=env_config['%d_%d_1'%(pp_deg, max_dp)]
            max_dp = max_dp // 2
        comm_coe_dict[pp_deg]['1']=0
        pp_deg *= 2
    return comm_coe_dict

def read_p2p_bandwidth_config(config_path, gpu_num):
    env_config = read_json_config(config_path)
    pp_deg = 2
    p2p_dict={}
    while pp_deg <= gpu_num:
        p2p_dict[pp_deg] = env_config['pp_deg_%d'%pp_deg]
        pp_deg *= 2
    return p2p_dict

def save_profiling_results(path, strategy, bsz, hidden_size, results):
    config = read_json_config(path) if os.path.exists(path) else {}
    key = form_strategy(strategy)
    if key not in config.keys():
        config[key] = {}
    config[key]['hidden%d_bsz%d'%(hidden_size, bsz)] = results
    write_json_config(config, path)
    print('Already written policy profiling config into config file %s!\n'%(path)) 

def save_profiled_memory(path, pp_deg, tp_deg, world_size, layer_num, bsz, rank, model_states, activation, activation_peak, cpt):
    config = read_json_config(path) if os.path.exists(path) else {}
    key = '%d_%d_%d'%(pp_deg,tp_deg,world_size//pp_deg//tp_deg)
    if cpt:
        key += '_c'
    if key not in config.keys():
        config[key] = {}
    if isinstance(layer_num, List):
        layernum_info = 'layernum[%s]'%(array2str(layer_num))
    else:
        layernum_info = 'layernum%d'%layer_num
    config[key]['%s_bsz%d_rank%d_ms'%(layernum_info, bsz, rank)] = model_states
    config[key]['%s_bsz%d_rank%d_act'%(layernum_info, bsz, rank)] = activation
    config[key]['%s_bsz%d_rank%d_act_peak'%(layernum_info, bsz, rank)] = activation_peak
    write_json_config(config, path)
    print('Already written profiled memory into config file %s!\n'%(path)) 