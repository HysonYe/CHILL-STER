import numpy as np
import copy

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

class AUVMobilityRMP:
    '''
    AUV mobility model that simulates motion using the Random Waypoint Model (RWP).
    '''
    def __init__(self, center_point, space, waypoints: list, padding = False, initial_pos = None, speed = 2.0):
        self._center_p = center_point
        self._space = space
        self._waypoints_list = copy.deepcopy(waypoints)
        self._waypoints = np.array(waypoints, dtype=float)
        self._speed = max(speed, 1e-6)
        self._padding = padding
        if initial_pos is None:
            self._initial_pos = self._waypoints[0] if len(self._waypoints) > 0 else np.array(center_point, dtype=float)
        else:
            self._initial_pos = np.array(initial_pos, dtype=float)
        self._target_waypoint = self._initial_pos
        self._pos = self._initial_pos
        self._cur_waypoint_idx = -1
        self._is_finished = False

    def reset(self):
        self._pos = self._initial_pos
        self._cur_waypoint_idx = -1
        self._is_finished = False
        self._setup_next_waypoint()

    def set_position(self, pos):
        self._pos = np.array(pos, dtype=float)

    def get_position(self):
        return list(self._pos)

    def get_speed(self):
        if self._speed < 1e-5:
            return 0.0
        return self._speed

    def set_speed(self, speed):
        self._speed = max(speed, 1e-6)

    def _gen_random_waypoint(self):
        space_l, space_w, space_d = self._space['x'], self._space['y'], self._space['z']
        rand_point = randpoint_on_space(self._center_p, space_l, space_w, space_d)
        return rand_point
    
    def _setup_next_waypoint(self, print_info = True):
        self._cur_waypoint_idx += 1
        if self._cur_waypoint_idx >= len(self._waypoints):
            if self._padding:
                rand_point = self._gen_random_waypoint()
                self._waypoints_list.append(rand_point)
                self._waypoints = np.array(self._waypoints_list, dtype=float)
            else:
                self._is_finished = True
                if print_info: print('[Motion] AUV has finished its mobility trajectory!')
                return
        self._target_waypoint = self._waypoints[self._cur_waypoint_idx]
        if print_info: print(f'[Motion] AUV moves to waypoint {self._cur_waypoint_idx}: {self._target_waypoint}')
    
    def _move(self, dt):
        if dt <= 0: return
        distance = np.linalg.norm(self._target_waypoint - self._pos)
        need_time = distance / self._speed
        if need_time <= dt:
            self._pos = self._target_waypoint
        else:
            direction = (self._target_waypoint - self._pos) / distance
            self._pos += direction * self._speed * dt

    def step(self, dt):
        if self._is_finished: return
        self._move(dt)
        if np.linalg.norm(self._target_waypoint - self._pos) < 1e-3:
            self._setup_next_waypoint()