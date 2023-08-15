import numpy as np
import torch

class MemoryCostModelDist:
    def __init__(self,
            strategy,
            global_batch_size = 8,
            parameter_size = 48,
            tp_activation_per_bsz_dict = {1:85, 2:47, 4:28, 8:18.5},
            other_memory_pp_off = {'model_states': 640, 'activation': 320},
            other_memory_pp_on = {'first_stage':{'model_states': 640, 'activation': 320}, 'last_stage':{'model_states': 640, 'activation': 320}},
            peak_reduction_with_chunks=None,
            microbatch=False,
            optimal_chunk_func=None,
            pytorch_context_mem = 1024,
            model_type='bert',
            checkpoint=0,
            use_zero2_for_dp=0,
            use_zero3_for_embed=0,
            mixed_precision=False,
            pipeline_type='gpipe',
            stage_idx=0,
            chunks=None):
        self.strategy = strategy
        self.pp_size = self.strategy[0]
        self.tp_size = self.strategy[1]
        self.dp_size = self.strategy[2]
        self.parameter_size = parameter_size/self.tp_size
        self.model_states_size = 4 * self.parameter_size

        self.bsz = global_batch_size/self.dp_size
        if chunks is None:
            chunks = optimal_chunk_func(global_batch_size//self.dp_size, strategy) if microbatch else 1
        max_chunks = global_batch_size // (self.tp_size*self.dp_size)
        max_chunks = 1 if max_chunks == 0 else max_chunks
        chunks = max_chunks if chunks > max_chunks else chunks
        chunks = int(chunks)
        if pipeline_type == 'pipedream_flush' and self.pp_size > 1:
            microbatches = [t.shape[0] for t in torch.arange(int(global_batch_size/self.dp_size/self.tp_size)).chunk(chunks)]
            chunks = len(microbatches)
            end = self.pp_size-stage_idx if self.pp_size-stage_idx <= chunks else chunks
            act_1f1b_ratio = np.sum(microbatches[:end]) / np.sum(microbatches)
            act_1f1b_ratio_first = np.sum(microbatches[:min(self.pp_size, chunks)]) / np.sum(microbatches)
            act_1f1b_ratio_last = microbatches[0] / np.sum(microbatches)
            self.bsz = act_1f1b_ratio * self.bsz
        
        if 'cpt' in self.strategy[-1].keys() and self.strategy[-1]['cpt']:
            assert(tp_activation_per_bsz_dict['checkpoint'] is not None)
            self.activation_size = tp_activation_per_bsz_dict['checkpoint'] * self.bsz
        else:
            self.activation_size = tp_activation_per_bsz_dict[self.tp_size] * self.bsz
        
        if chunks == 1:
            zero2_ratio = lambda d: (7/8 * (1/d + 0.003) + 1/8) if mixed_precision else lambda d: (3/4 * (1/d + 0.003) + 1/4)
            zero3_ratio = lambda d: (1/d+0.003)
        else:
            zero2_ratio = lambda d: (6/8 * (1/d + 0.003) + 2/8) if mixed_precision else lambda d: (2/4 * (1/d + 0.003) + 2/4)
            zero3_ratio = lambda d: (7/8 * (1/d + 0.003) + 1/8) if mixed_precision else lambda d: (3/4 * (1/d + 0.003) + 1/4)
        
        if 'fsdp' in self.strategy[-1].keys() and self.strategy[-1]['fsdp']:
            # fsdp_model_states memory is slightly larger than dp_model_states/dp_size
            # we add a small bias to ensure the predicted fsdp memory NOT smaller than real value
            # Actually, this bias barely affect search result.
            self.model_states_size  *= zero3_ratio(self.dp_size)
        elif 'fsdp' in self.strategy[-1].keys() and self.strategy[-1]['fsdp']==0 and use_zero2_for_dp:
            self.model_states_size *= zero2_ratio(self.dp_size)
        
        self.total = self.model_states_size + self.activation_size
        self.other_memcosts = [0] * self.pp_size
        other_layers_bsz = global_batch_size/self.tp_size/self.dp_size
        
        other_ms_zero2_ratio = zero3_ratio(self.tp_size*self.dp_size) if use_zero3_for_embed else (zero2_ratio(self.tp_size*self.dp_size) if use_zero2_for_dp else 1.0)
        
        if self.pp_size == 1:
            self.other_memcosts[0] += other_memory_pp_off['model_states'] * other_ms_zero2_ratio + other_memory_pp_off['activation'] * other_layers_bsz
        else:
            if pipeline_type == 'pipedream_flush':
                other_layers_bsz_first = other_layers_bsz * act_1f1b_ratio_first
                other_layers_bsz_last = other_layers_bsz * act_1f1b_ratio_last
            else:
                other_layers_bsz_first = other_layers_bsz_last = other_layers_bsz
            # Model type may affect other memory performance (embedding, cls, etc.)
            if model_type in ['bert', 't5']:
                self.other_memcosts[0] += other_memory_pp_on['first_stage']['model_states'] * other_ms_zero2_ratio + other_memory_pp_on['first_stage']['activation'] * (other_layers_bsz_first/self.pp_size)
            elif model_type in ['vit', 'swin', 'gpt']:
                self.other_memcosts[0] += other_memory_pp_on['first_stage']['model_states'] * other_ms_zero2_ratio + other_memory_pp_on['first_stage']['activation'] * other_layers_bsz_first
                # When chunks get larger, peak memory may reduce. Adjust peak memory if needed.
                if peak_reduction_with_chunks is not None: 
                    if isinstance(peak_reduction_with_chunks, dict):
                        self.other_memcosts[0] -= peak_reduction_with_chunks['first'] * other_layers_bsz_first * (1 - 1 / chunks)
                    else:
                        self.other_memcosts[0] -= peak_reduction_with_chunks * other_layers_bsz_first * (1 - 1 / chunks)
            if model_type in ['swin']:
                self.other_memcosts[-1] += other_memory_pp_on['last_stage']['model_states'] * other_ms_zero2_ratio + other_memory_pp_on['last_stage']['activation'] * (other_layers_bsz_last/self.pp_size)
            elif model_type in ['bert', 't5', 'vit', 'gpt']:
                self.other_memcosts[-1] += other_memory_pp_on['last_stage']['model_states'] * other_ms_zero2_ratio + other_memory_pp_on['last_stage']['activation'] * other_layers_bsz_last
                # When chunks get larger, peak memory may reduce. Adjust peak memory if needed.
                if peak_reduction_with_chunks is not None: 
                    if isinstance(peak_reduction_with_chunks, dict):
                        self.other_memcosts[-1] -= peak_reduction_with_chunks['last'] * other_layers_bsz_last * (1 - 1 / chunks)
                    else:
                        self.other_memcosts[-1] -= peak_reduction_with_chunks * other_layers_bsz_last * (1 - 1 / chunks)

        if checkpoint:
            for i in range(len(self.other_memcosts)):
                self.other_memcosts[i] += tp_activation_per_bsz_dict[1] * self.bsz // self.tp_size

        for i in range(len(self.other_memcosts)):
            self.other_memcosts[i] += pytorch_context_mem

    def get_memory_cost(self):
        result = dict()
        result['parameter'] = self.parameter_size
        result['model_states'] = self.model_states_size
        result['activation'] = self.activation_size
        result['enc_total'] = self.total
        result['other'] = self.other_memcosts
        return result


class TimeCostModelDist_with_overlap:
    def __init__(self,
            strategy,
            global_batch_size,
            parameter_size = 48,
            microbatch=True,
            optimal_chunk_func = None,
            sequence_length=512,
            hidden_size=1024,
            forward_computation_time=35 / 24,
            bct_fct_coe=2,
            extra_overhead=0,
            comm_coe_dict={},
            dp_overlap_coe=1.3,
            bct_overlap_coe=1.3,
            p2p_comm_coe_dict=None,
            layer_num=None,
            layer_type='enc',
            use_zero2_for_dp=0,
            mixed_precision=False):
        self.s = strategy[:3]
        self.sl = sequence_length
        self.hs = hidden_size
        self.microbatch = microbatch
        self.pp_size = self.s[0]
        self.tp_size = self.s[1]
        self.dp_size = self.s[2]
        self.comm_coe_dict = comm_coe_dict[self.pp_size]
        if self.tp_size == 1 or self.dp_size == 1:
            self.dc = self.comm_coe_dict['%d'%self.dp_size]
            self.tc = self.comm_coe_dict['%d'%self.tp_size]
        else:
            # In this case, strategy[-1]['tp'] represents tp_consecutive_flag
            info = strategy[-1]
            assert 'tp' in info.keys() and info['tp'] in [0, 1]
            tp_consecutive_flag = info['tp']
            if tp_consecutive_flag:
                self.dc = self.comm_coe_dict['%d_0'%self.dp_size]
                self.tc = self.comm_coe_dict['%d_1'%self.tp_size]
            else:
                self.dc = self.comm_coe_dict['%d_1'%self.dp_size]
                self.tc = self.comm_coe_dict['%d_0'%self.tp_size]
        self.fsdp = False
        if 'fsdp' in strategy[-1].keys() and strategy[-1]['fsdp']:
            self.fsdp = True
        self.dp_overlap_coe = dp_overlap_coe
        self.dc_overlap = self.dc*dp_overlap_coe
        self.ps = parameter_size/self.tp_size
        self.bs = global_batch_size/self.dp_size 
        self.layer_type = layer_type
        assert(layer_type in ['enc', 'dec'])
        self.optimal_microbatch = optimal_chunk_func(self.bs, self.s) if microbatch else 1

        # Dummy layer_num, can be any multiple of 8.
        # We estimate the time cost of single layer by averaging the time of whole model to deal with pipeline parallel
        self.layer_num = 24 if layer_num is None else layer_num

        self.checkpoint = False
        if 'cpt' in strategy[-1].keys() and strategy[-1]['cpt']:
            self.checkpoint = True

        # forward & backward computation time of whole model (depending on dummy layer_num)
        self.fct = forward_computation_time * self.bs / self.tp_size * self.layer_num 
        self.bct = self.fct * bct_fct_coe
        self.bct_overlap_coe = bct_overlap_coe
        self.bct_overlap = self.bct*bct_overlap_coe
        self.eo = extra_overhead

        # dp & tp message size of whole model (depending on dummy layer_num)
        self.dp_message_size = (2*(self.dp_size-1)/self.dp_size*self.ps) * self.layer_num
        tp_comm_times = 4 if layer_type=='enc' else 6
        self.tp_message_size = 2*(self.tp_size-1)/self.tp_size*(self.bs*self.sl*self.hs*tp_comm_times*4/1024/1024) * self.layer_num

        if self.fsdp:
            self.dp_message_size = self.dp_message_size * 0.5

        self.p2p_comm_coe = None
        if self.pp_size > 1 and p2p_comm_coe_dict is not None:
            self.p2p_comm_coe = p2p_comm_coe_dict[self.pp_size]
            self.p2p_meg_size = 2*self.bs*self.sl*self.hs*4/1024/1024

        self.use_zero2_for_dp = use_zero2_for_dp
        if self.checkpoint:
            self.fct *= 2
            self.tp_message_size *= 1.5

        if mixed_precision:
            self.dp_message_size = self.dp_message_size/2
            self.tp_message_size = self.tp_message_size/2

    def bct_dp_overlap(self, dp_message_size, bct):
        dp_overlap_time = dp_message_size * self.dc_overlap
        bct_overlap_time = bct * self.bct_overlap_coe
        if dp_overlap_time > bct_overlap_time:
            overlap_part = bct_overlap_time
            rest_part = (dp_message_size - bct_overlap_time / self.dc_overlap) * self.dc
            rest_dp_flag = True
        elif dp_overlap_time < bct_overlap_time:
            overlap_part = dp_overlap_time
            rest_part = (bct - dp_overlap_time / self.bct_overlap_coe) 
            rest_dp_flag = False
        else:
            overlap_part = bct_overlap_time
            rest_part = 0
            rest_dp_flag = False
        rest_dp_flag = False
        return overlap_part, rest_part, rest_dp_flag

    def pipe_with_microbatch(self, computation_overhead, communication_overhead):
        result = computation_overhead*(self.pp_size+self.optimal_microbatch-1)/(self.pp_size*self.optimal_microbatch)+communication_overhead
        return result

    def gen_result(self):
        if self.pp_size == 1:
            if self.tp_size == 1 and self.dp_size > 1: # pure dp
                overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
                result = self.fct + overlap_part + rest_part + self.eo
            elif self.dp_size == 1 and self.tp_size > 1: # pure tp
                result = self.fct + self.bct + self.tp_message_size*self.tc
            else: # dp+tp
                if self.tp_size < self.tp_size * self.dp_size // 2: 
                    if self.layer_type == 'enc':
                        overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
                        result = self.fct + overlap_part + rest_part + self.tp_message_size*self.tc + self.eo
                    elif self.layer_type == 'dec':
                        overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct*2/3)
                        result = self.fct + 1/3*self.bct + overlap_part + rest_part +self.tp_message_size*self.tc+self.eo
                else:
                    if self.layer_type == 'enc':
                        overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct*1/2)
                        result = self.fct + 1/2*self.bct + overlap_part + rest_part + self.tp_message_size*self.tc + self.eo
                    elif self.layer_type == 'dec':
                        overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct*2/3)
                        result = self.fct + 1/3*self.bct + overlap_part + rest_part + self.tp_message_size*self.tc + self.eo
        elif self.pp_size > 1:
            if self.tp_size == 1 and self.dp_size > 1: # pp+dp
                overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
                overall_overhead = self.fct + overlap_part + rest_part + self.eo
                if self.microbatch == False:
                    result = overall_overhead
                else:
                    computation_overhead = self.fct + self.bct
                    communication_overhead = overall_overhead-computation_overhead
                    result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
            elif self.dp_size == 1 and self.tp_size > 1: # pp+tp
                if self.microbatch == False:
                    result = self.fct + self.bct + self.tp_message_size*self.tc
                else:
                    overall_overhead = self.fct + self.bct + self.tp_message_size*self.tc
                    result = self.pipe_with_microbatch(overall_overhead, 0)
            elif self.dp_size == 1 and self.tp_size == 1: # pure pp
                if self.microbatch == False:
                    result = self.fct + self.bct
                else:
                    overall_overhead = self.fct + self.bct
                    result = self.pipe_with_microbatch(overall_overhead, 0)
            else: # pp+dp+tp
                if self.tp_size < self.tp_size * self.dp_size // 2:
                    if self.layer_type == 'enc':
                        overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
                        overall_overhead = self.fct + overlap_part + rest_part + self.tp_message_size*self.tc + self.eo
                    elif self.layer_type == 'dec':
                        overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct*2/3)
                        overall_overhead = self.fct + 1/3*self.bct + overlap_part + rest_part +self.tp_message_size*self.tc+self.eo
                    if self.microbatch == False:
                        result = overall_overhead
                    else:
                        computation_overhead = self.fct + self.bct + self.tp_message_size*self.tc
                        communication_overhead = overall_overhead-computation_overhead
                        result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
                else:
                    if self.layer_type == 'enc':
                        overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct*1/2)
                        overall_overhead = self.fct + 1/2*self.bct + overlap_part + rest_part + self.tp_message_size*self.tc + self.eo
                    elif self.layer_type == 'dec':
                        overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct*2/3)
                        overall_overhead = self.fct + 1/3*self.bct + overlap_part + rest_part + self.tp_message_size*self.tc + self.eo
                    if self.microbatch == False:
                        result = overall_overhead
                    else:
                        computation_overhead = self.fct + self.bct + self.tp_message_size*self.tc
                        communication_overhead = overall_overhead-computation_overhead
                        result = self.pipe_with_microbatch(computation_overhead, communication_overhead)



        # For fsdp, add allgather time of forward and backward
        if self.fsdp:
            forward_allgather_time = self.dp_message_size * self.dc 
            # if self.checkpoint:
            #     forward_allgather_time *= 2
            backward_allgather_time = self.dp_message_size * self.dc 
            result = result + (forward_allgather_time + backward_allgather_time)*self.optimal_microbatch

        if self.pp_size > 1 and self.p2p_comm_coe is not None:
            result = result + self.p2p_meg_size * self.p2p_comm_coe
        
        coe = 0.00085
        result = result*coe
        result = result / self.layer_num
        return result
    
def check_optimal_chunks(world_size, strategies, optimal_chunk_func, bsz):
    chunk_dict = {}
    for pp_deg in sorted(set([s[0] for s in strategies])):
        chunk_dict[pp_deg] = optimal_chunk_func(bsz/(world_size//pp_deg), [pp_deg,1,world_size//pp_deg,{'fsdp':0,'cpt':0}])
    return chunk_dict