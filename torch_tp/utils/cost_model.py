import numpy as np

class MemoryCostModel:
    def __init__(self,
            strategy,
            global_batch_size = 8,
            parameter_size = 48,
            tp_activation_per_bsz_dict = {1:85, 2:47, 4:28, 8:18.5},
            other_model_states = 640, 
            other_activation_per_bsz = 320,
            pytorch_context_mem = 1024):
        self.strategy = strategy
        self.pp_size = self.strategy[0]
        self.tp_size = self.strategy[1]
        self.dp_size = self.strategy[2]
        self.parameter_size = parameter_size/self.tp_size
        self.model_states_size = 4 * self.parameter_size
        if 'fsdp' in self.strategy[-1].keys() and self.strategy[-1]['fsdp']:
            # fsdp_model_states memory is slightly larger than dp_model_states/dp_size
            # we add a small bias to ensure the predicted fsdp memory NOT smaller than real value
            # Actually, this bias barely affect search result.
            self.model_states_size  *= (1/self.dp_size + 0.025)
        self.bsz = global_batch_size/self.dp_size
        self.activation_size = tp_activation_per_bsz_dict[self.tp_size] * self.bsz
        self.total = self.model_states_size + self.activation_size
        self.other_memcost = other_model_states + other_activation_per_bsz * (global_batch_size/self.tp_size/self.dp_size)
        self.other_memcost += pytorch_context_mem

    def get_memory_cost(self):
        result = dict()
        result['parameter'] = self.parameter_size
        result['model_states'] = self.model_states_size
        result['activation'] = self.activation_size
        result['enc_total'] = self.total
        result['other'] = self.other_memcost
        return result

class TimeCostModel_without_overlap:
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
            extra_overhead=80,
            comm_coe_dict={1:{'8':5/24, '4_0':5/24, '4_1':6/24, '2_0':5/24, '2_1':6/24, '1':0}, 
                            2:{'4':5/24, '2_0':5/24, '2_1':6/24, '1':0}, 
                            4:{'2':5/24, '1':0}, 
                            8:{'1':0}},
            layer_type='enc'):
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
        self.ps = parameter_size/self.tp_size
        self.bs = global_batch_size/self.dp_size
        # Dummy layer_num, can be any multiple of 8.
        # We estimate the time cost of single layer by averaging the time of whole model to deal with pipeline parallel
        self.layer_num = 24 
        assert(layer_type in ['enc', 'dec'])
        if microbatch:
            self.optimal_microbatch = optimal_chunk_func(self.bs, self.s)
        self.fct = forward_computation_time * self.bs / self.tp_size * self.layer_num 
        self.bct = self.fct * bct_fct_coe
        self.eo = extra_overhead
        self.dp_message_size = (2*(self.dp_size-1)/self.dp_size*self.ps) * self.layer_num
        tp_comm_times = 4 if layer_type=='enc' else 6
        self.tp_message_size = 2*(self.tp_size-1)/self.tp_size*(self.bs*self.sl*self.hs*tp_comm_times*4/1024/1024) * self.layer_num

    def pipe_with_microbatch(self, computation_overhead, communication_overhead):
        result = computation_overhead*(self.pp_size+self.optimal_microbatch-1)/(self.pp_size*self.optimal_microbatch)+communication_overhead
        return result

    def gen_result(self):
        computation_overhead = self.fct+self.bct
        if self.pp_size == 8:
            communication_overhead = 0
        elif self.dp_size == 1 and self.tp_size != 1:
            communication_overhead = self.tc*self.tp_message_size
        elif self.dp_size != 1 and self.tp_size == 1:
            communication_overhead = self.dc*self.dp_message_size/self.pp_size+self.eo
        else:
            communication_overhead = self.tc*self.tp_message_size+self.dc*self.dp_message_size/self.pp_size+self.eo
        if self.microbatch == False:
            result = computation_overhead + communication_overhead
        else:
            result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
        coe = 0.0011 / self.layer_num
        result = result*coe
        return result


class TimeCostModel_with_overlap:
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
            layer_type='enc'):
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
        if microbatch:
            self.optimal_microbatch = optimal_chunk_func(self.bs, self.s)
            # print(self.optimal_microbatch)

        # Dummy layer_num, can be any multiple of 8.
        # We estimate the time cost of single layer by averaging the time of whole model to deal with pipeline parallel
        self.layer_num = 24 

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
        return overlap_part, rest_part, rest_dp_flag

    def pipe_with_microbatch(self, computation_overhead, communication_overhead):
        result = computation_overhead*(self.pp_size+self.optimal_microbatch-1)/(self.pp_size*self.optimal_microbatch)+communication_overhead
        return result

    def gen_result(self):
        # print(self.dp_message_size, self.tp_message_size/2)
        # print('dp:',self.dp_message_size*self.dc, 'bct:',self.bct, 'tp_bwd:',self.tp_message_size*self.tc/2)
        if np.array_equal(self.s, [1,1,8]) == True:
            overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
            result = self.fct + overlap_part + rest_part + self.eo
        elif np.array_equal(self.s, [1,2,4]) == True:
            if self.layer_type == 'enc':
                overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
                result = self.fct + overlap_part + rest_part + self.tp_message_size*self.tc + self.eo
            elif self.layer_type == 'dec':
                overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct*2/3)
                result = self.fct + 1/3*self.bct + overlap_part + rest_part +self.tp_message_size*self.tc+self.eo
        elif np.array_equal(self.s, [1,4,2]) == True:
            if self.layer_type == 'enc':
                overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct*1/2)
                result = self.fct + 1/2*self.bct + overlap_part + rest_part + self.tp_message_size*self.tc + self.eo
            elif self.layer_type == 'dec':
                overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct*2/3)
                result = self.fct + 1/3*self.bct + overlap_part + rest_part + self.tp_message_size*self.tc + self.eo
        elif np.array_equal(self.s, [1,8,1]) == True:
            result = self.fct + self.bct + self.tp_message_size*self.tc
        elif np.array_equal(self.s, [2,1,4]) == True:
            bct_per_stage = self.bct / self.pp_size
            dp_message_size_per_stage = self.dp_message_size / self.pp_size
            overlap_part_per_stage, rest_part_per_stage, rest_dp_flag = self.bct_dp_overlap(dp_message_size_per_stage, bct_per_stage)
            # print(overlap_part_per_stage, rest_part_per_stage, rest_dp_flag)
            if rest_dp_flag and not self.fsdp:
                overall_overhead = self.fct + overlap_part_per_stage * self.pp_size + rest_part_per_stage + self.eo
            else:
                overall_overhead = self.fct + (overlap_part_per_stage + rest_part_per_stage) * self.pp_size + self.eo
            if self.microbatch == False:
                result = overall_overhead
            else:
                computation_overhead = self.fct + self.bct
                communication_overhead = overall_overhead-computation_overhead
                result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
        elif np.array_equal(self.s, [2,2,2]) == True:
            overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct*1/2)
            overall_overhead = self.fct + overlap_part + rest_part + self.bct*1/2 + self.tp_message_size*self.tc + self.eo
            if self.microbatch == False:
                result = overall_overhead
            else:
                computation_overhead = self.fct + self.bct + self.tp_message_size*self.tc
                communication_overhead = overall_overhead-computation_overhead
                result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
        elif np.array_equal(self.s, [2,4,1]) == True:
            if self.microbatch == False:
                result = self.fct + self.bct + self.tp_message_size*self.tc
            else:
                overall_overhead = self.fct + self.bct + self.tp_message_size*self.tc
                result = self.pipe_with_microbatch(overall_overhead, 0)
        elif np.array_equal(self.s, [4,1,2]) == True:
            # new version
            bct_per_stage = self.bct / self.pp_size
            dp_message_size_per_stage = self.dp_message_size / self.pp_size
            overlap_part_per_stage, rest_part_per_stage, rest_dp_flag = self.bct_dp_overlap(dp_message_size_per_stage, bct_per_stage)
            # print(overlap_part_per_stage, rest_part_per_stage, rest_dp_flag)
            if rest_dp_flag and not self.fsdp:
                overall_overhead = self.fct + overlap_part_per_stage * self.pp_size + rest_part_per_stage + self.eo
            else:
                overall_overhead = self.fct + (overlap_part_per_stage + rest_part_per_stage) * self.pp_size + self.eo

            # # old version
            # overall_overhead = self.fct+self.dp_message_size*self.dc_overlap/4+(self.bct_overlap-self.dp_message_size*self.dc_overlap/4)/self.bct_overlap_coe+self.eo

            if self.microbatch == False:
                result = overall_overhead
            else:
                computation_overhead = self.fct + self.bct
                communication_overhead = overall_overhead-computation_overhead
                result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
        elif np.array_equal(self.s, [4,2,1]) == True:
            if self.microbatch == False:
                result = self.fct + self.bct + self.tp_message_size*self.tc
            else:
                overall_overhead = self.fct + self.bct + self.tp_message_size*self.tc
                result = self.pipe_with_microbatch(overall_overhead, 0)
        elif np.array_equal(self.s, [8,1,1]) == True:
            if self.microbatch == False:
                result = self.fct + self.bct
            else:
                overall_overhead = self.fct + self.bct
                result = self.pipe_with_microbatch(overall_overhead, 0)
        if self.fsdp:
            # print(result, self.dp_message_size * 0.5 * self.dc)
            result = result + self.dp_message_size * 0.5 * self.dc
        coe = 0.0011
        result = result*coe
        result = result / self.layer_num
        return result

class TimeCostModel_with_overlap_old:
    def __init__(self,
            strategy,
            global_batch_size,
            parameter_size = 48,
            microbatch=True,
            optimal_chunk_func = None,
            sequence_length=512,
            hidden_size=1024,
            forward_computation_time=35,
            bct_fct_coe=2,
            extra_overhead=80,
            comm_coe_dict={1:{'8':5, '4_0':5, '4_1':6, '2_0':5, '2_1':6, '1':0}, 
                            2:{'4':5, '2_0':5, '2_1':6, '1':0}, 
                            4:{'2':5, '1':0}, 
                            8:{'1':0}},
            dp_overlap_coe=1.3,
            bct_overlap_coe=1.3,
            layer_type='enc'):
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
        self.dco = self.dc*dp_overlap_coe
        self.ps = parameter_size/self.tp_size
        self.bs = global_batch_size/self.dp_size 
        self.layer_type = layer_type
        assert(layer_type in ['enc', 'dec'])
        if microbatch:
            self.optimal_microbatch = optimal_chunk_func(self.bs, self.s)
        # Dummy layer_num, can be any multiple of 8.
        # We estimate the time cost of single layer by averaging the time of whole model to deal with pipeline parallel
        self.layer_num = 24 
        self.fct = forward_computation_time*(global_batch_size//8) * self.layer_num 
        self.bct = forward_computation_time*bct_fct_coe*(global_batch_size//8) * self.layer_num 
        self.bct_overlap_coe = bct_overlap_coe
        self.bco = self.bct*bct_overlap_coe
        self.eo = extra_overhead
        self.dp_message_size = (2*(self.dp_size-1)/self.dp_size*self.ps)/self.pp_size * self.layer_num 
        # if self.fsdp:
        #     self.dp_message_size = self.dp_message_size * 1.5
        tp_comm_times = 4 if layer_type=='enc' else 6
        self.tp_message_size = 2*(self.tp_size-1)/self.tp_size*(self.bs*self.sl*self.hs*tp_comm_times*4/1024/1024) * self.layer_num 

    def overlap(self, type):
        if type == 0:
            if self.dp_message_size > self.bco/self.dco:
                result = self.bco+(self.dp_message_size-self.bco/self.dco)*self.dc
            elif self.dp_message_size < self.bco/self.dco:
                result = (self.bco-self.dp_message_size*self.dco)/self.bct_overlap_coe+self.dp_message_size*self.dco
            else:
                result = self.bco
        else:
            if self.layer_type == 'enc':
                # result = self.bct+5/7*self.dp_message_size*self.dc
                result = self.bct+max(self.dp_message_size*self.dc-self.bct*1/2, 0)
            elif self.layer_type == 'dec':
                result = self.bct+max(self.dp_message_size*self.dc-self.bct*2/3, 0)
        return result
    
    def pipe_with_microbatch(self, computation_overhead, communication_overhead):
        result = computation_overhead*(self.pp_size+self.optimal_microbatch-1)/(self.pp_size*self.optimal_microbatch)+communication_overhead
        return result

    def gen_result(self):
        if np.array_equal(self.s, [1,1,8]) == True:
            result = self.fct+self.overlap(0)+self.eo
        elif np.array_equal(self.s, [1,2,4]) == True:
            if self.layer_type == 'enc':
                result = self.fct+self.overlap(0)+self.tp_message_size*self.tc+self.eo
            elif self.layer_type == 'dec':
                bct_non_overlap = 1/3 * self.bct
                self.bct = 2/3 * self.bct
                self.bco = 2/3 * self.bco
                result = self.fct+self.overlap(0)+self.tp_message_size*self.tc+self.eo+bct_non_overlap
        elif np.array_equal(self.s, [1,4,2]) == True:
            result = self.fct+self.tp_message_size*self.tc+self.overlap(1)+self.eo
        elif np.array_equal(self.s, [1,8,1]) == True:
            result = self.fct+self.bct+self.tp_message_size*self.tc
        elif np.array_equal(self.s, [2,1,4]) == True:
            if self.microbatch == False:
                result = self.fct*2+self.bco+self.overlap(0)+self.eo
            else:
                overall_overhead = self.fct*2+self.bco+self.overlap(0)+self.eo
                computation_overhead = self.fct*2+self.bct*2
                communication_overhead = overall_overhead-computation_overhead
                result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
        elif np.array_equal(self.s, [2,2,2]) == True:
            if self.microbatch == False:
                self.dp_message_size=2*self.dp_message_size
                result = self.fct*2+self.overlap(0)+self.bct+self.tp_message_size*self.tc+self.eo
            else:
                self.dp_message_size=2*self.dp_message_size
                overall_overhead = self.fct*2+self.overlap(0)+self.bct+self.tp_message_size*self.tc+self.eo
                computation_overhead = self.fct*2+self.bct*2+self.tp_message_size*self.tc
                communication_overhead = overall_overhead-computation_overhead
                result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
        elif np.array_equal(self.s, [2,4,1]) == True:
            if self.microbatch == False:
                result = self.fct*2+self.bct*2+self.tp_message_size*self.tc
            else:
                overall_overhead = self.fct*2+self.bct*2+self.tp_message_size*self.tc
                result = self.pipe_with_microbatch(overall_overhead, 0)
        elif np.array_equal(self.s, [4,1,2]) == True:
            if self.microbatch == False:
                result = self.fct*4+self.dp_message_size*self.dco+(self.bco*4-self.dp_message_size*self.dco)/self.bct_overlap_coe+self.eo
            else:
                overall_overhead = self.fct*4+self.dp_message_size*self.dco+(self.bco*4-self.dp_message_size*self.dco)/self.bct_overlap_coe+self.eo
                computation_overhead = self.fct*4+self.bct*4
                communication_overhead = overall_overhead-computation_overhead
                result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
        elif np.array_equal(self.s, [4,2,1]) == True:
            if self.microbatch == False:
                result = self.fct*4+self.bct*4+self.tp_message_size*self.tc
            else:
                overall_overhead = self.fct*4+self.bct*4+self.tp_message_size*self.tc
                result = self.pipe_with_microbatch(overall_overhead, 0)
        elif np.array_equal(self.s, [8,1,1]) == True:
            if self.microbatch == False:
                result = self.fct*8+self.bct*8
            else:
                overall_overhead = self.fct*8+self.bct*8
                result = self.pipe_with_microbatch(overall_overhead, 0)

        # if self.fsdp:
        #     result = result + self.dp_message_size * 0.5 * self.dc
        coe = 0.0012 / self.layer_num 
        result = result*coe

        # if self.fsdp:
        #     # bert huge
        #     if np.array_equal(self.s, [1,1,8]):
        #         result += 0.34565 / 24
        #     elif np.array_equal(self.s, [1,2,4]):
        #         result += 0.25615 / 24
        #     elif np.array_equal(self.s, [1,4,2]):
        #         result += 0.1531 / 24
        #     elif np.array_equal(self.s, [2,1,4]):
        #         result += 0.4557 / 24
        #     elif np.array_equal(self.s, [2,2,2]):
        #         result += 0.22985 / 24
        #     elif np.array_equal(self.s, [4,1,2]):
        #         result += 0.2138 / 24

            # # vit huge
            # if np.array_equal(self.s, [1,1,8]):
            #     result += 0.2706 / 24
            # elif np.array_equal(self.s, [1,2,4]):
            #     result += 0.2274 / 24
            # elif np.array_equal(self.s, [1,4,2]):
            #     result += 0.2065 / 24
            # elif np.array_equal(self.s, [2,1,4]):
            #     result += 0.5351 / 24
            # elif np.array_equal(self.s, [2,2,2]):
            #     result += 0.2297 / 24
            # elif np.array_equal(self.s, [4,1,2]):
            #     result += 0.2087 / 24
        return result

# if __name__ == '__main__':
#     # test microbatch
#     strategies = [[2,1,4],[2,2,2],[2,4,1],[4,1,2],[4,2,1],[8,1,1]]
#     global_batch = [8,16,24,32,40]
#     for i in range(6):
#         print('LOG: ', strategies[i])
#         for j in range(5):
#             result = TimeCostModel_with_overlap(strategies[i], global_batch[j], 48, microbatch=True).gen_result()
#             print(result)
#         print()

#     strategies = [[1,1,8],[1,2,4],[1,4,2],[1,8,1],[2,1,4],[2,2,2],[2,4,1],[4,1,2],[4,2,1],[8,1,1]]
#     for i in range(10):
#         result = TimeCostModel_with_overlap(strategies[i], 8, 48).gen_result()
#         print(result)
#     print()
#     for i in range(10):
#         result = TimeCostModel_with_overlap(strategies[i], 16, 48).gen_result()
#         print(result)
#     print()
#     for i in range(10):
#         result = TimeCostModel_with_overlap(strategies[i], 24, 48).gen_result()
#         print(result)
#     print()
#     for i in range(10):
#         result = TimeCostModel_with_overlap(strategies[i], 32, 48).gen_result()
#         print(result)
#     print()

#     for i in range(10):
#         result = TimeCostModel_without_overlap(strategies[i], 8, 48).gen_result()
#         print(result)
#     print()
#     for i in range(10):
#         result = TimeCostModel_without_overlap(strategies[i], 16, 48).gen_result()
#         print(result)
#     print()
#     for i in range(10):
#         result = TimeCostModel_without_overlap(strategies[i], 24, 48).gen_result()
#         print(result)
#     print()
#     for i in range(10):
#         result = TimeCostModel_without_overlap(strategies[i], 32, 48).gen_result()
#         print(result)
#     print()

#     tp_activation_per_bsz_dict = {  1:85.00833, 
#                                     2:47.00833, 
#                                     4:28.008125, 
#                                     8:18.54875}
#     other_model_states = 640
#     other_activation_per_bsz = 320
#     for i in range(10):
#         result = MemoryCostModel(strategies[i], 8, 48.05, tp_activation_per_bsz_dict, other_model_states, other_activation_per_bsz).get_memory_cost()
#         print(result['enc_total'], result['other'], result['enc_total']*24/strategies[i][0]+result['other'])
#     print()
#     for i in range(10):
#         result = MemoryCostModel(strategies[i], 16, 48.05, tp_activation_per_bsz_dict, other_model_states, other_activation_per_bsz).get_memory_cost()
#         print(result['enc_total'], result['other'], result['enc_total']*24/strategies[i][0]+result['other'])
#     print()
#     for i in range(10):
#         result = MemoryCostModel(strategies[i], 24, 48.05, tp_activation_per_bsz_dict, other_model_states, other_activation_per_bsz).get_memory_cost()
#         print(result['enc_total'], result['other'], result['enc_total']*24/strategies[i][0]+result['other'])
#     print()
#     for i in range(10):
#         result = MemoryCostModel(strategies[i], 32, 48.05, tp_activation_per_bsz_dict, other_model_states, other_activation_per_bsz).get_memory_cost()
#         print(result['enc_total'], result['other'], result['enc_total']*24/strategies[i][0]+result['other'])
#     print()
