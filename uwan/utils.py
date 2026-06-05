import math, random
import numpy as np
from infra.utils.tools import up_to_integer
from typing import Tuple
from infra.environment import COORD_ORDER

def get_delay_slot( pos_1: Tuple[float, float, float],
                    pos_2: Tuple[float, float, float],
                    prop_sped: float,
                    slot_size: float) -> float:
    distance = math.dist(pos_1, pos_2)
    delay = distance / prop_sped
    delay_slot_float = delay / slot_size
    delay_slot = up_to_integer(delay_slot_float)

    return delay_slot

def randpoint_on_sphere_z_ge0(
        center: Tuple[float, float, float],
        dist: float,
    ) -> Tuple[float, float, float]:
    '''
    Randomly sample a point on a sphere surface centered at 'center' with radius 'dist', 
    requiring the z-coordinate to be greater than or equal to 0.

    Args:
        center (Tuple[float, float, float]): Coordinates of the sphere center.
        dist (float): Radius of the sphere, must be positive.

    Returns:
        Tuple[float, float, float]: Coordinates of the randomly sampled point on the sphere surface.
    '''
    if dist <= 0: raise ValueError("[Error] The dist must be positive.")
    x_c, y_c, z_c = center
    cos_theta_min = max(-1.0, -z_c / dist)
    if cos_theta_min > 1.0:
        raise ValueError("[Error] The entire sphere lies below z = 0; no point can satisfy z >= 0.")

    u = np.random.uniform(cos_theta_min, 1.0)
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    sin_theta = np.sqrt(1.0 - u * u)
    unit_vec = np.array([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        u
    ])

    # Move to target sphere surface
    center = np.asarray(center, dtype=float)
    pos_2 = center + dist * unit_vec
    if pos_2[2] < 0: pos_2[2] = 0.0

    point = [round(v,3) for v in pos_2]
    return tuple(point)

def randopint_on_cuboid(center, l_x, l_y, l_z):
    '''
    Randomly sample a point inside a cuboid centered at 'center' with side lengths 'l_x', 'l_y', and 'l_z'.
    
    The z-axis range is restricted to [10, l_z-10] to avoid being too close to the water surface or bottom.

    Args:
        center (Tuple[float, float, float]): Center coordinates of the cuboid.
        l_x (float): Side length of the cuboid along the x-axis, must be positive.
        l_y (float): Side length of the cuboid along the y-axis, must be positive.
        l_z (float): Side length of the cuboid along the z-axis, must be positive.
    '''
    margin = 20
    cx, cy, _ = center
    min_x = cx - l_x / 2 + margin
    max_x = cx + l_x / 2 - margin
    min_y = cy - l_y / 2 + margin
    max_y = cy + l_y / 2 - margin
    min_z = 10
    max_z = l_z - 10
    current_x = int(random.uniform(min_x, max_x))
    current_y = int(random.uniform(min_y, max_y))
    current_z = int(random.uniform(min_z, max_z))
    return (current_x, current_y, current_z)

def randpoint_on_space(center, length, width, depth):
    '''
    Randomly sample a point inside a specified 3D rectangular space.

    Args:
        center (Tuple[float, float, float]): Center coordinates of the space (x, y, z).
        length (float): Length of the space along the x-axis.
        width (float): Width of the space along the y-axis.
        depth (float): Depth extent along the positive z direction from the center's z.

    Returns:
        Tuple[float, float, float]: Coordinates of the sampled point (rounded to 3 decimals).
    '''
    if length <= 0 or width <= 0 or depth <= 0:
        raise ValueError("length, width, and depth must be positive numbers.")
    
    x = np.random.uniform(center[0] - length / 2, center[0] + length / 2)
    y = np.random.uniform(center[1] - width / 2, center[1] + width / 2)
    z = np.random.uniform(center[2], center[2] + depth)

    return (round(x,3), round(y,3), round(z,3))

def unpack_place_info(place, slot_size, prop_speed, AP_pos, space):
    '''
    Unpack the node placement dictionary, returning the delay and position coordinates.

    Args:
        place (dict): Node placement dictionary containing delay and position.
            - delay (int): Propagation delay.
            - coord (list[float]): Position coordinates [x, y, z].
        slot_size (int): Slot size in seconds.
        prop_speed (float): Underwater acoustic propagation speed in m/s.
        AP_pos (Tuple[float, float, float]): Position coordinates of the Access Point.
        space (dict): Water space size, containing ranges for 'x', 'y', and 'z' dimensions.
    
    Returns:
        delay (int): Propagation delay in terms of slots between the node and the AP.
        pos (dict): Node position coordinate dictionary containing 'x', 'y', and 'z'.
    '''
    _d, _p = place['delay'], place['coord']
    delay = -1
    pos = {
        'x': 0.0,
        'y': 0.0,
        'z': 0.0
    }
    if len(_p) == 3:
        for i, xyz in enumerate(COORD_ORDER):
            pos[xyz] = _p[i]
        delay = get_delay_slot(AP_pos, _p, prop_speed, slot_size)
    elif _d > 0:
        delay = _d
        _p = randpoint_on_sphere_z_ge0(AP_pos, _d * slot_size * prop_speed)
        for i, xyz in enumerate(COORD_ORDER):
            pos[xyz] = round(_p[i], 3)
    else:
        _p = randopint_on_cuboid(AP_pos, space['x'], space['y'], space['z'])
        delay = get_delay_slot(AP_pos, _p, prop_speed, slot_size)
        for i, xyz in enumerate(COORD_ORDER):
            pos[xyz] = round(_p[i], 3)
    return delay, pos

def allocate_tdma_slots(nodes):
    """
    Allocate collision-free slot strategies for TDMA nodes.

    Args:
        nodes (list of dict): A list of dictionaries, each containing:
            - "id": Unique identifier for the node.
            - "strategy": If slots are already allocated, a list of length 'frame_size' with elements 0 or 1; 
                          if not, [n_slots_needed, frame_size].
            - "delay": Propagation delay in slots from node to AP.
    """
    if not nodes:
        return []
    
    # Verify frame size consistency
    frame_size = None
    for node in nodes:
        strategy = node["strategy"]
        if len(strategy) == 2:
            current_fs = strategy[1]
        elif len(strategy) > 2:
            current_fs = len(strategy)
        else:
            raise ValueError("Unrecognized strategy format.")

        if frame_size is None:
            frame_size = current_fs
        elif frame_size != current_fs:
            raise ValueError(f"Error: Inconsistent frame sizes among nodes! Found {frame_size} and {current_fs}.")

    # Identify occupied slots
    ap_timeline = [False] * frame_size
    unassigned_nodes = []
    for node in nodes:
        strategy = node["strategy"]
        delay = node["delay"]
        if len(strategy) > 2:   # Nodes with pre-allocated slots
            for send_slot, is_sending in enumerate(strategy):
                if is_sending == 1:
                    ap_arrival_slot = (send_slot + delay) % frame_size
                    if ap_timeline[ap_arrival_slot]:
                        raise ValueError("Error: Existing node strategies cause collisions at the AP!")
                    ap_timeline[ap_arrival_slot] = True
        else:
            unassigned_nodes.append(node)
    
    # Calculate total demand for remaining nodes and check for available slots
    total_needed_slots = sum(node["strategy"][0] for node in unassigned_nodes)
    available_slots = ap_timeline.count(False)
    if total_needed_slots > available_slots:
         raise ValueError(f"Error: Total node demand ({total_needed_slots}) exceeds remaining available slots ({available_slots}). Allocation failed!")
         
    # Start slot allocation
    free_ap_slots = [i for i, is_occupied in enumerate(ap_timeline) if not is_occupied]
    random.shuffle(free_ap_slots)  # Shuffle available slots to increase randomness in allocation
    free_slot_index = 0
    for node in unassigned_nodes:
        slots_needed = node["strategy"][0]
        delay = node["delay"]
        new_strategy = [0] * frame_size
        for _ in range(slots_needed):
            # Select an idle target slot at the AP
            target_ap_slot = free_ap_slots[free_slot_index]
            free_slot_index += 1
            
            # Calculate the corresponding sending slot
            send_slot = (target_ap_slot - delay) % frame_size
            # Set the corresponding sending slot to 1
            new_strategy[send_slot] = 1

        node["strategy"] = new_strategy
        
    return nodes
