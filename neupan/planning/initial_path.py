"""
Initial path generation from waypoints (line, Dubins, Reeds-Shepp curves).
"""

import math
import numpy as np
from math import tan, inf, cos, sin, sqrt
from gctl import curve_generator
from neupan.utils.math import WrapToPi, distance


class InitialPath:
    """Generate and track the naive initial path for MPC.

    Args:
        receding: MPC horizon.  step_time: MPC dt.
        ref_speed: Reference speed.  robot: Robot instance.
        waypoints: [[x,y,yaw], ...].  loop: Loop back to start.
        curve_style: 'line' | 'dubins' | 'reeds'.
    """

    def __init__(self, receding, step_time, ref_speed, robot,
                 waypoints=None, loop=False, curve_style="line", **kw):
        self.T, self.dt, self.ref_speed, self.robot = receding, step_time, ref_speed, robot
        self.waypoints = self._to_np_list(waypoints)
        self.loop, self.curve_style = loop, curve_style
        self.min_radius = kw.get("min_radius", self._default_turn_radius())
        self.interval = kw.get("interval", self.dt * self.ref_speed)
        self.arrive_threshold = kw.get("arrive_threshold", 0.1)
        self.close_threshold = kw.get("close_threshold", 0.1)
        self.ind_range = kw.get("ind_range", 10)
        self.arrive_index_threshold = kw.get("arrive_index_threshold", 1)
        self.arrive_flag = False
        self.cg = curve_generator()
        self.initial_path = None

    # ==== public API ====

    def generate_nom_ref_state(self, state, cur_vel_array, ref_speed):
        state = state[:3]
        ref_state = self.cur_point[0:3].copy()
        ref_index = self.point_index
        pre_state = state.copy()
        state_pre_list, state_ref_list = [pre_state], [ref_state]
        assert self.cur_point.shape[0] >= 4
        gear_list = [self.cur_point[-1, 0]] * self.T
        rsf = ref_speed * self.dt

        for t in range(self.T):
            pre_state = self._motion_predict(pre_state, cur_vel_array[:, t:t+1], self.robot.L, self.dt)
            state_pre_list.append(pre_state)
            if rsf >= self.interval:
                ref_index = ref_index + int(rsf / self.interval)
                if ref_index > len(self.cur_curve) - 1:
                    ref_index = len(self.cur_curve) - 1; gear_list[t] = 0
                ref_state = self.cur_curve[ref_index][0:3]
            else:
                ref_state, ref_index = self._find_interaction_point(ref_state, ref_index, rsf)
                if ref_index > len(self.cur_curve) - 1:
                    gear_list[t] = 0
            diff = ref_state[2, 0] - pre_state[2, 0]
            ref_state[2, 0] = pre_state[2, 0] + WrapToPi(diff)
            state_ref_list.append(ref_state)

        return np.hstack(state_pre_list), cur_vel_array, np.hstack(state_ref_list), np.array(gear_list) * ref_speed

    def check_arrive(self, state):
        self.init_check(state)
        self.closest_point(state, self.close_threshold, self.ind_range)
        if self._check_curve_arrive(state, self.arrive_threshold, self.arrive_index_threshold):
            if self.curve_index + 1 >= self.curve_number:
                if self.loop:
                    self.curve_index = self.point_index = 0
                    print("Info: loop, reset the path"); return False
                if not self.arrive_flag:
                    print("Info: arrive at the end of the path"); self.arrive_flag = True
                return True
            else:
                self.curve_index += 1; self.point_index = 0
        return False

    def set_initial_path(self, path):
        self.initial_path = path
        self.interval = self._cal_average_interval(path)
        self._split_path_with_gear()
        self.curve_index = self.point_index = 0

    def set_ipath_with_state(self, state):
        self._init_path_with_state(state[0:3])
        self._split_path_with_gear()
        self.curve_index = self.point_index = 0

    def set_ipath_with_waypoints(self, waypoints):
        self.initial_path = self.cg.generate_curve(self.curve_style, waypoints, self.interval, self.min_radius, True)
        if self.curve_style == "line": self._ensure_consistent_angles()
        self._split_path_with_gear()
        self.curve_index = self.point_index = 0
        self.waypoints = waypoints

    def update_initial_path_from_goal(self, start, goal):
        wps = [start, goal, start] if self.loop else [start, goal]
        self.initial_path = self.cg.generate_curve(self.curve_style, wps, self.interval, self.min_radius, True)
        if self.curve_style == "line": self._ensure_consistent_angles()
        self._split_path_with_gear()
        self.curve_index = self.point_index = 0
        self.waypoints = wps

    def init_check(self, state):
        if self.initial_path is None:
            print("initial path not set — generating from current state")
            self.set_ipath_with_state(state)

    def reset(self):
        self.point_index = self.curve_index = 0
        self.arrive_flag = False

    # ---- properties ----
    @property
    def cur_waypoints(self): return self.waypoints
    @property
    def cur_curve(self): return self.curve_list[self.curve_index]
    @property
    def cur_point(self): return self.cur_curve[self.point_index]
    @property
    def curve_number(self): return len(self.curve_list)

    # ==== internal ====

    def closest_point(self, state, threshold=0.1, ind_range=10):
        min_dis, start = inf, max(self.point_index, 0)
        for idx in range(start, min(self.point_index + ind_range, len(self.cur_curve))):
            d = distance(state[0:2], self.cur_curve[idx][0:2])
            if d < min_dis:
                min_dis = d; self.point_index = idx
                if d < threshold: break
        return min_dis

    def _find_interaction_point(self, ref_state, ref_index, length):
        circle = np.squeeze(ref_state[0:2])
        while True:
            if ref_index > len(self.cur_curve) - 2:
                ep = self.cur_curve[-1]; ep[2] = WrapToPi(ep[2])
                return ep[0:3], ref_index
            cur, nxt = self.cur_curve[ref_index], self.cur_curve[ref_index + 1]
            seg = [np.squeeze(cur[0:2]), np.squeeze(nxt[0:2])]
            ip = self._range_cir_seg(circle, length, seg)
            if ip is not None:
                diff = WrapToPi(nxt[2, 0] - cur[2, 0])
                theta = WrapToPi(cur[2, 0] + diff / 2)
                return np.append(ip, theta).reshape(3, 1), ref_index
            ref_index += 1

    @staticmethod
    def _range_cir_seg(circle, r, segment):
        sp, ep = segment
        d = ep - sp
        if np.linalg.norm(d) == 0: return None
        f = sp - circle
        a, b, c = d @ d, 2 * f @ d, f @ f - r ** 2
        disc = b ** 2 - 4 * a * c
        if disc < 0: return None
        t2 = (-b + sqrt(disc)) / (2 * a)
        return sp + t2 * d if 0 <= t2 <= 1 else None

    def _check_curve_arrive(self, state, threshold=0.1, idx_threshold=2):
        d = np.linalg.norm(state[0:2] - self.cur_curve[-1][0:2])
        return d < threshold and self.point_index >= len(self.cur_curve) - idx_threshold - 2

    def _split_path_with_gear(self):
        self.curve_list, cur, gear = [], [], self.initial_path[0][-1]
        for pt in self.initial_path:
            if pt[-1] != gear:
                self.curve_list.append(cur); cur = []; gear = pt[-1]
            cur.append(pt)
        if cur: self.curve_list.append(cur)

    def _init_path_with_state(self, state):
        assert len(self.waypoints) > 0
        self.waypoints = ([state] + self.waypoints if isinstance(self.waypoints, list)
                          else np.vstack([state, self.waypoints]))
        if self.loop: self.waypoints = self.waypoints + [self.waypoints[0]]
        self.initial_path = self.cg.generate_curve(self.curve_style, self.waypoints, self.interval, self.min_radius, True)
        if self.curve_style == "line": self._ensure_consistent_angles()

    def _cal_average_interval(self, path):
        if len(path) < 2: return 0
        return sum(math.hypot(*(b[0:2] - a[0:2]).flat) for a, b in zip(path, path[1:])) / (len(path) - 1)

    def _ensure_consistent_angles(self):
        if not self.initial_path or len(self.initial_path) < 2: return
        for i in range(len(self.initial_path) - 1):
            c, n = self.initial_path[i], self.initial_path[i + 1]
            c[2, 0] = math.atan2(n[1, 0] - c[1, 0], n[0, 0] - c[0, 0])
        self.initial_path[-1][2, 0] = self.initial_path[-2][2, 0]

    def _motion_predict(self, s, v, L, dt):
        k = self.robot.kinematics
        if k == "acker":  return self._acker(s, v, L, dt)
        if k == "diff":   return self._diff(s, v, dt)
        if k == "omni":   return self._omni(s, v, dt)

    @staticmethod
    def _acker(s, v, L, dt):
        phi, spd, psi = s[2, 0], v[0, 0], v[1, 0]
        return s + np.array([[spd*cos(phi)], [spd*sin(phi)], [spd*tan(psi)/L]]) * dt

    @staticmethod
    def _diff(s, v, dt):
        phi, spd, w = s[2, 0], v[0, 0], v[1, 0]
        return s + np.array([[spd*cos(phi)], [spd*sin(phi)], [w]]) * dt

    @staticmethod
    def _omni(s, v, dt):
        vx, vy = v[0, 0]*cos(v[1, 0]), v[0, 0]*sin(v[1, 0])
        return s + np.array([[vx], [vy], [0]]) * dt

    def _default_turn_radius(self):
        if self.robot.kinematics == "acker":
            return self.robot.L / tan(self.robot.max_speed[1])
        return 0.0

    @staticmethod
    def _to_np_list(pts):
        return [] if pts is None else [np.c_[p] if isinstance(p, list) else p for p in pts]
