import math
import torch
import numpy as np

def _distribute_side(total: int, n: int) -> list[int]:
    if n == 0: return []
    denom = (1 << n) - 1
    weights = [2 ** (n - i - 1) for i in range(n)]
    alloc = [total * w // denom for w in weights]
    residue = total - sum(alloc)
    i = 0
    while residue:
        alloc[i] += 1
        residue  -= 1
        i = (i + 1) if i + 1 < n else 0
    return alloc

def _get_base_sampling_batchs(cur_task_idx: int, total_tasks: int, batch_size: int = 64, sampling_tasks: int = 3) -> list[int]:
    '''
    Calculates the distribution of samples across tasks based on the current task index,
    total task count, batch size, and sampling window size.

    Args:
        cur_task_idx (int): The index of the current task, in the range [0, total_tasks - 1].
        total_tasks (int): Total number of tasks, must be positive.
        batch_size (int): Total batch size to distribute, must be positive.
        sampling_tasks (int): Size of the sampling window (must be an odd number).
    
    Returns:
        res (list[int]): A list of length `total_tasks` where each element represents 
            the number of samples allocated to the corresponding task.
    '''
    if sampling_tasks % 2 == 0:
        raise ValueError("The 'sampling_tasks' must be an odd number.")
    if total_tasks <= 0:
        raise ValueError("The value of 'total_tasks' must be greater than 0.")
    if batch_size <= 0:
        return [0] * total_tasks
    if total_tasks == 1:
        return [batch_size]

    k = sampling_tasks // 2
    center_cnt = batch_size // 2
    leftover = batch_size - center_cnt
    left_total = leftover // 2
    right_total = leftover - left_total

    left_cap  = min(k, cur_task_idx)
    right_cap = min(k, total_tasks - cur_task_idx - 1)

    if left_cap == 0:
        right_total += left_total
        left_total = 0
    if right_cap == 0:
        left_total += right_total
        right_total = 0
        right_cap = 0

    left_near = _distribute_side(left_total,  left_cap)     # Implements a distribution strategy following the "distance-decay principle".
    right_near = _distribute_side(right_total, right_cap)

    res = [0] * total_tasks
    res[cur_task_idx] = center_cnt
    for d, cnt in enumerate(left_near, start=1):
        res[cur_task_idx - d] = cnt
    for d, cnt in enumerate(right_near, start=1):
        res[cur_task_idx + d] = cnt

    return res

def get_context_batchs(batch_size: int, spatial_context: float, spatial_size: int, radius: int = 3) -> list[int]:
    x_float = float(spatial_context)        # Convert to float for calculation
    if x_float < 0:                         # Handle boundary cases
        x_float = 0.0
    elif x_float > spatial_size - 1:
        x_float = float(spatial_size - 1)
    
    # Integer position: directly compute sampling batches
    if x_float.is_integer():
        return _get_base_sampling_batchs(int(x_float), spatial_size, batch_size, radius)
    
    # Non-integer position
    i = math.floor(x_float)                 # Integer part
    batchs_i = _get_base_sampling_batchs(i, spatial_size, batch_size, radius)
    batchs_j = _get_base_sampling_batchs(i + 1, spatial_size, batch_size, radius)
    
    alpha = x_float - i                     # Fractional part for linear interpolation
    float_batchs = (1 - alpha)*np.array(batchs_i) + alpha*np.array(batchs_j)
    base_batchs = [int(round(x,0)) for x in float_batchs]

    return base_batchs

def up_to_integer(value: float) -> int:
    frac, int_part = math.modf(value)
    if frac < 1e-3:
        _integer = int(int_part)
    else:
        _integer = int(int_part) + 1
    return _integer

def get_autocast_dtype(device_name):
    if 'cuda' in device_name.lower() and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:
            return torch.bfloat16
        else:
            return torch.float16
    else:
        return torch.float32

def ripple_traverse(arr, idx):
    if not arr: return []
    if idx < 0 or idx >= len(arr): raise IndexError("Index out of array bounds")
    result = [arr[idx]]
    l, r = idx - 1, idx + 1

    while l >= 0 or r < len(arr):
        if l >= 0:
            result.append(arr[l])
            l -= 1
        if r < len(arr):
            result.append(arr[r])
            r += 1
    return result