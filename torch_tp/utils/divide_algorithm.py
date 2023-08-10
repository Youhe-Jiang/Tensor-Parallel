import heapq
import torch

def partition_balanced(weights, pipeline_parallel_size):
    num_total = pipeline_parallel_size
    num_items = len(weights)
    length = len(weights)
    num = num_total
    prefix = [1 if w == 0 else w for w in weights]
    for i in range(1, length):
        prefix[i] += prefix[i - 1]

    lower_bound = max(weights)
    upper_bound = prefix[length - 1]

    while upper_bound > lower_bound:
        mid = (upper_bound + lower_bound) // 2
        prev_ = 0
        prefix_ = 0
        num_block = 0
    
        for idx, w in enumerate(prefix):
            if prefix[idx] - prefix_ > mid:
                prev_ = idx
                prefix_ = prefix[idx - 1]
                num_block += 1
        number = num_block + 1
        if number <= num:
            upper_bound = mid
        else:
            lower_bound = mid + 1

    prev_ = 0
    prefix_ = 0
    num_block = 0
    intervals = []

    for idx, w in enumerate(prefix):
        if prefix[idx] - prefix_ > upper_bound:
            intervals.append((prev_, idx))
            prev_ = idx
            prefix_ = prefix[idx - 1]
            num_block += 1

    intervals.append((prev_, len(prefix)))
    num_block += 1
    if num_block < num:
        def _heap_push(heap, st, ed):
            value = weights[ed - 1]
            if st > 0:
                value -= weights[st - 1]
            heapq.heappush(heap, (-value, st, ed))
        
        ret_intervals = []
        heap = []
        add_cnt = num - num_block
        weights = prefix
        for st, ed in intervals:
            _heap_push(heap, st, ed)

        while add_cnt > 0:
            _, st, ed = heapq.heappop(heap)
            if ed - st == 1:
                ret_intervals.append((st, ed))
            else:
                w_sum = weights[ed - 1]
                prefix = 0
                if st > 0:
                    w_sum -= weights[st - 1]
                    prefix = weights[st - 1]
                minimum = float("inf")
                for idx in range(st + 1, ed):
                    front = weights[idx - 1] - prefix
                    diff = abs(w_sum - 2 * front)
                    if diff < minimum:
                        pos = idx
                        minimum = diff 
                l, m, r = st, pos, ed
                _heap_push(heap, l, m)
                _heap_push(heap, m, r)
                add_cnt -= 1

        while heap:
            _, st, ed = heapq.heappop(heap)
            ret_intervals.append((st, ed))

        ret_intervals.sort()
        intervals = ret_intervals
    
    current = 0
    parts = []
    for inter in intervals:
        parts.append(inter[1]-inter[0])
        current = (current + 1) % pipeline_parallel_size

    return parts

# Example:
# weights = [50] * 24 + [50] * 24
# parts = partition_balanced(weights, 8)
# print('model weights:', weights)
# print('partition method:', parts)
