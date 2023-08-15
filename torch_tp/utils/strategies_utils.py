def generate_strategies(gpu_num, type):
    i = 1
    total = []
    while i<=gpu_num:
        total.append(i)
        i *= 2
    if type == 'full':
        strategies = []
        for pp in total:
            for tp in total:
                if pp*tp<=gpu_num:
                    dp = gpu_num // (pp * tp) 
                    if tp==1 or tp == gpu_num/pp:
                        if dp == 1:
                            strategies.append([pp,tp,dp,{}])
                        else:
                            strategies.append([pp,tp,dp,{'fsdp':0}])
                            strategies.append([pp,tp,dp,{'fsdp':1}])
                    else:
                        strategies.append([pp,tp,dp,{'tp':0,'fsdp':0}])
                        strategies.append([pp,tp,dp,{'tp':0,'fsdp':1}])
                        strategies.append([pp,tp,dp,{'tp':1,'fsdp':0}])
                        strategies.append([pp,tp,dp,{'tp':1,'fsdp':1}])
        return strategies
    elif type == 'dp+tp':
        strategies = []
        pp = 1
        for tp in total:
            if pp*tp<=gpu_num:
                dp = gpu_num // (pp * tp) 
                if tp==1 or tp == gpu_num/pp:
                    if dp == 1:
                        strategies.append([pp,tp,dp,{}])
                    else:
                        strategies.append([pp,tp,dp,{'fsdp':0}])
                        # strategies.append([pp,tp,dp,{'fsdp':1}])
                else:
                    strategies.append([pp,tp,dp,{'tp':0,'fsdp':0}])
                    # strategies.append([pp,tp,dp,{'tp':0,'fsdp':1}])
                    strategies.append([pp,tp,dp,{'tp':1,'fsdp':0}])
                    # strategies.append([pp,tp,dp,{'tp':1,'fsdp':1}])
        return strategies
    elif type == 'dp+pp':
        strategies = []
        tp = 1
        for pp in total:
            if pp*tp<=gpu_num:
                dp = gpu_num // (pp * tp) 
                if tp==1 or tp == gpu_num/pp:
                    if dp == 1:
                        strategies.append([pp,tp,dp,{}])
                    else:
                        strategies.append([pp,tp,dp,{'fsdp':0}])
                        # strategies.append([pp,tp,dp,{'fsdp':1}])
                else:
                    strategies.append([pp,tp,dp,{'tp':0,'fsdp':0}])
                    # strategies.append([pp,tp,dp,{'tp':0,'fsdp':1}])
                    strategies.append([pp,tp,dp,{'tp':1,'fsdp':0}])
                    # strategies.append([pp,tp,dp,{'tp':1,'fsdp':1}])
        return strategies