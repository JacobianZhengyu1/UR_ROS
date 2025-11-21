import numpy as np
import yaml
from math import pi
from collections import defaultdict
try:
    from numba import njit
except:

    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap


@njit(fastmath=True, cache=True)
def _inv_se3_njit(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Rt = R.T
    Ti[:3, :3] = Rt
    Ti[:3, 3] = Ti[:3, 3] = -np.dot(np.ascontiguousarray(Rt), np.ascontiguousarray(t)) # -np.dot(Rt, t) # -Rt @ t
    return Ti

@njit(fastmath=True, cache=True)
def _DH2tform_forward_njit(alpha, a, d, theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.empty((4, 4), dtype=np.float64)
    T[0, 0] = ct
    T[0, 1] = -st * ca
    T[0, 2] = st * sa
    T[0, 3] = a * ct

    T[1, 0] = st
    T[1, 1] = ct * ca
    T[1, 2] = -ct * sa
    T[1, 3] = a * st

    T[2, 0] = 0.0
    T[2, 1] = sa
    T[2, 2] = ca
    T[2, 3] = d

    T[3, 0] = 0.0
    T[3, 1] = 0.0
    T[3, 2] = 0.0
    T[3, 3] = 1.0
    return T

@njit(fastmath=True, cache=True)
def _DH2tform_inverse_njit(alpha, a, d, theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.empty((4, 4), dtype=np.float64)
    T[0, 0] = ct
    T[0, 1] = -st
    T[0, 2] = 0
    T[0, 3] = a

    T[1, 0] = st * ca
    T[1, 1] = ct * ca
    T[1, 2] = -sa
    T[1, 3] = -sa * d

    T[2, 0] = st * sa
    T[2, 1] = ct * sa
    T[2, 2] = ca
    T[2, 3] = ca * d

    T[3, 0] = 0.0
    T[3, 1] = 0.0
    T[3, 2] = 0.0
    T[3, 3] = 1.0
    return T

@njit(fastmath=True, cache=True)
def _forward_dh_njit(thetas, d, a, alpha):
    T = np.eye(4, dtype=np.float64)
    for i in range(6):
        Ti = _DH2tform_forward_njit(alpha[i], a[i], d[i], thetas[i])
        T = np.dot(T, Ti)
    return T

@njit(fastmath=True, cache=True)
def _euler_zyx_to_R(zyx):
    z, y, x = zyx[0], zyx[1], zyx[2]
    cz, sz = np.cos(z), np.sin(z)
    cy, sy = np.cos(y), np.sin(y)
    cx, sx = np.cos(x), np.sin(x)
    Rz = np.array([[cz, -sz, 0.0],
                   [sz,  cz, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0, cy]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cx, -sx],
                   [0.0, sx,  cx]], dtype=np.float64)
    return Rz @ Ry @ Rx

@njit(fastmath=True, cache=True)
def _angle_wrap_pi(a):
    out = a
    while out <= -np.pi:
        out += 2.0*np.pi
    while out > np.pi:
        out -= 2.0*np.pi
    return out

@njit(fastmath=True, cache=True)
def _feasibility_forward_compare(config, d, a, alpha, T06, tol):
    Tfk = _forward_dh_njit(config, d, a, alpha)
    diff = np.abs(Tfk - T06).mean()
    return diff < tol



class Kinematics:
    def __init__(self, arm='ur5', file_path=None, optimized=True):

        self.optimized = True
        self.current_joint_angles = None

        self.optimized = optimized
    
        if not self.optimized:
            
            try:
                with open(file_path, 'r') as f: # type: ignore
                    self.configs = yaml.safe_load(f)
                #print("YAML data loaded successfully:")
            except FileNotFoundError:
                print(f"Error: The file '{file_path}' was not found.")
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file: {e}")

            self.current_joint_angles = None

            self.arm_config = self.configs[arm]
            self.dh_parameters = self.arm_config['dh']

            self.a1 = self.dh_parameters['a1']
            self.a2 = self.dh_parameters['a2']
            self.a3 = self.dh_parameters['a3']
            self.a4 = self.dh_parameters['a4']
            self.a5 = self.dh_parameters['a5']
            self.a6 = self.dh_parameters['a6']

            self.d1 = self.dh_parameters['d1']
            self.d2 = self.dh_parameters['d2']
            self.d3 = self.dh_parameters['d3']
            self.d4 = self.dh_parameters['d4']
            self.d5 = self.dh_parameters['d5']
            self.d6 = self.dh_parameters['d6']

            self.alpha1 = self.dh_parameters['alpha1']
            self.alpha2 = self.dh_parameters['alpha2']
            self.alpha3 = self.dh_parameters['alpha3']
            self.alpha4 = self.dh_parameters['alpha4']
            self.alpha5 = self.dh_parameters['alpha5']
            self.alpha6 = self.dh_parameters['alpha6']

        else: 
            if arm == 'ur5e':
                self.a1 = 0.0
                self.a2 = -0.425
                self.a3 = -0.3922
                self.a4 = 0.0
                self.a5 = 0.0
                self.a6 = 0.0

                self.d1 = 0.1625
                self.d2 = 0.0
                self.d3 = 0.0
                self.d4 = 0.1333
                self.d5 = 0.0997
                self.d6 = 0.0996

                self.alpha1 = 1.5707963267948966
                self.alpha2 = 0.0
                self.alpha3 = 0.0
                self.alpha4 = 1.5707963267948966
                self.alpha5 = -1.5707963267948966
                self.alpha6 = 0.0

            elif arm == 'ur5':
                self.a1 = 0.0
                self.a2 = -0.425
                self.a3 = -0.39225
                self.a4 = 0.0
                self.a5 = 0.0
                self.a6 = 0.0

                self.d1 = 0.089159
                self.d2 = 0.0
                self.d3 = 0.0
                self.d4 = 0.10915
                self.d5 = 0.09465
                self.d6 = 0.0823

                self.alpha1 = 1.5707963267948966
                self.alpha2 = 0.0
                self.alpha3 = 0.0
                self.alpha4 = 1.5707963267948966
                self.alpha5 = -1.5707963267948966
                self.alpha6 = 0.0
            else:
                raise NameError(f"DH parameters of the specified Arm ({arm}) is not defined!")

        self.d = np.array([self.d1, self.d2, self.d3, self.d4, self.d5, self.d6], dtype=np.float64)
        self.a = np.array([self.a1, self.a2, self.a3, self.a4, self.a5, self.a6], dtype=np.float64)
        self.alpha = np.array([self.alpha1, self.alpha2, self.alpha3, self.alpha4, self.alpha5, self.alpha6], dtype=np.float64)

    # ------------------ Forward ------------------

    def forward(self, thetas, degree=False):
        thetas = np.array(thetas, dtype=np.float64).reshape(6)
        if degree:
            thetas = np.deg2rad(thetas)
        return _forward_dh_njit(thetas, self.d, self.a, self.alpha)

    # ------------------ Inverse ------------------

    def inverse(self, ee_pose):
        
        T06 = ee_pose
        if T06.shape == (3, 4):
            T06 = np.vstack((T06, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)))
        T06 = T06.astype(np.float64, copy=False)
        # print('T06: ', T06)
        d1, d4, d5, d6 = self.d1, self.d4, self.d5, self.d6
        a2, a3 = self.a2, self.a3
        a1, a4, a5, a6 = self.a1, self.a4, self.a5, self.a6
        alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = self.alpha

        P06 = T06[:3, 3]
        P05 = (np.dot(T06, np.array([0.0, 0.0, -d6, 1.0], dtype=np.float64)))[:3]

        # -------- theta1 (دو جواب) --------
        th1 = np.full(2, np.nan, dtype=np.float64)
        base = np.arctan2(P05[1], P05[0]) + np.pi/2.0
        r = np.hypot(P05[0], P05[1])
        cval = d4 / max(r, 1e-12)
        cval = np.clip(cval, -1.0, 1.0)
        phi = np.arccos(cval)
        th1[0] = _angle_wrap_pi(base - phi)
        th1[1] = _angle_wrap_pi(base + phi)
        
        # print("th1:    ", th1)

        # -------- theta5 (چهار جواب) --------
        th5 = np.full(4, np.nan, dtype=np.float64)
        for i in range(2):
            s1, c1 = np.sin(th1[i]), np.cos(th1[i])
            numer = (P06[0]*s1 - P06[1]*c1 - d4)
            val = numer / d6
            val = np.clip(val, -1.0, 1.0)
            th5[2*i + 0] = np.arccos(val)
            th5[2*i + 1] = -np.arccos(val)
            
        # print("th5:    ", th5)

        # -------- theta6 (چهار جواب) --------
        th6 = np.full(4, np.nan, dtype=np.float64)
        T60 = _inv_se3_njit(T06)
        # print('T60: ', T60)
        X60 = T60[:3, 0]
        Y60 = T60[:3, 1]
        for i in range(2):
            s1, c1 = np.sin(th1[i]), np.cos(th1[i])
            for j in range(2):
                t5 = th5[2*i + j]
                s5 = np.sin(t5)
                if np.abs(s5) < 1e-10:
                    th6[2*i + j] = np.nan
                    continue
                left = (-X60[1]*s1 + Y60[1]*c1) / s5
                right = ( X60[0]*s1 - Y60[0]*c1) / s5
                th6[2*i + j] = np.arctan2(left, right)

        # print("th6:    ", th6)
        
        # -------- P14 و theta3 (هشت جواب) --------
        def _calc_P14(theta1, theta5, theta6):
            T01 = _DH2tform_inverse_njit(0.0, 0.0, d1, theta1)
            T10 = _inv_se3_njit(T01)

            T45 = _DH2tform_inverse_njit(alpha4, a4, d5, theta5)
            T54 = _inv_se3_njit(T45)

            T56 = _DH2tform_inverse_njit(alpha5, a5, d6, theta6)
            T65 = _inv_se3_njit(T56)

            T14 = T10 @ T06 @ T65 @ T54
            return T14[:3, 3], T14

        th3 = np.full(8, np.nan, dtype=np.float64)
        signs = np.array([+1.0, -1.0], dtype=np.float64)

        for i in range(8):
            t1 = th1[i // 4]
            t5 = th5[(i // 2)]
            t6 = th6[(i // 2)]
            if not (np.isfinite(t1) and np.isfinite(t5) and np.isfinite(t6)):
                continue
            # print('data:    ',t1, t5, t6)
            P14, _ = _calc_P14(t1, t5, t6)
            # print('P14: ', P14)
            r_xz = np.hypot(P14[0], P14[2])
            lo = np.abs(a2 - a3)
            hi = np.abs(a2 + a3)
            if not (r_xz > lo - 1e-12 and r_xz < hi + 1e-12):
                continue
            cos_t3 = (r_xz*r_xz - a2*a2 - a3*a3) / (2.0*a2*a3)
            cos_t3 = np.clip(cos_t3, -1.0, 1.0)
            t3_mag = np.arccos(cos_t3)
            th3[i] = signs[i % 2] * t3_mag


        # print("th3:    ", th3)
        # -------- theta2 (هشت جواب) --------
        th2 = np.full(8, np.nan, dtype=np.float64)
        for i in range(8):
            t1 = th1[i // 4]
            t5 = th5[(i // 2)]
            t6 = th6[(i // 2)]
            t3 = th3[i]
            if not (np.isfinite(t1) and np.isfinite(t5) and np.isfinite(t6) and np.isfinite(t3)):
                continue
            P14, _ = _calc_P14(t1, t5, t6)
            r_xz = np.hypot(P14[0], P14[2])
            if r_xz < 1e-12:
                continue
            # arctan2(-z, -x) - arcsin(-a3*sin(t3)/r)
            theta2 = np.arctan2(-P14[2], -P14[0]) - np.arcsin(np.clip(-a3*np.sin(t3)/r_xz, -1.0, 1.0))
            th2[i] = _angle_wrap_pi(theta2)

        # print("th2:    ", th2)
        # -------- theta4 (هشت جواب) --------
        th4 = np.full(8, np.nan, dtype=np.float64)

        def _calc_theta4(t1, t2, t3, t5, t6):
            P14, T14 = _calc_P14(t1, t5, t6)
            T12 = _DH2tform_inverse_njit(self.alpha1, self.a1, self.d2, t2)
            T21 = _inv_se3_njit(T12)
            T23 = _DH2tform_inverse_njit(self.alpha2, self.a2, self.d3, t3)
            T32 = _inv_se3_njit(T23)
            T34 = T32 @ T21 @ T14
            X34 = T34[:3, 0]
            return np.arctan2(X34[1], X34[0])

        for i in range(8):
            t1 = th1[i // 4]
            t2 = th2[i]
            t3 = th3[i]
            t5 = th5[(i // 2)]
            t6 = th6[(i // 2)]
            if not (np.isfinite(t1) and np.isfinite(t2) and np.isfinite(t3) and np.isfinite(t5) and np.isfinite(t6)):
                continue
            th4[i] = _calc_theta4(t1, t2, t3, t5, t6)

        # -------- مرتب‌سازی و بررسی صحت --------
        def _sorted_solution(th1_, th2_, th3_, th4_, th5_, th6_):
            configs = defaultdict(list)
            for i in range(8):
                t1 = th1_[i // 4]
                t2 = th2_[i]
                t3 = th3_[i]
                t4 = th4_[i]
                t5 = th5_[(i // 2)]
                t6 = th6_[(i // 2)]
                configs[str(i)] = [t1, t2, t3, t4, t5, t6]

            # همان ترتیب نسخه‌ی قبلی شما
            order = ['3', '7', '1', '5', '0', '4', '6', '2']
            return [np.array(configs[k], dtype=np.float64) for k in order]

        configs = _sorted_solution(th1, th2, th3, th4, th5, th6)

        feas = np.zeros(8, dtype=bool)
        feasible_configs = []
        T06f = T06.astype(np.float64)
        for i, cfg in enumerate(configs):
            if np.any(~np.isfinite(cfg)):
                feas[i] = False
                continue
            ok = _feasibility_forward_compare(cfg, self.d, self.a, self.alpha, T06f, 1e-4)
            feas[i] = ok
            if ok:
                feasible_configs.append(cfg)

        fc = np.array(feasible_configs, dtype=np.float64) if feasible_configs else np.empty((0, 6), dtype=np.float64)
        return feas, configs, fc

    # ------------------ ساختن پوژ ۴×۴ ------------------

    def make_ee_pose(self, position_vector, degree=False):
        pv = np.array(position_vector, dtype=np.float64).reshape(-1)
        if pv.size < 6:
            tmp = np.zeros(6, dtype=np.float64)
            tmp[:pv.size] = pv
            pv = tmp
        elif pv.size > 6:
            pv = pv[:6]

        t = pv[:3].reshape(3)
        rpy = pv[3:].reshape(3)
        if degree:
            rpy = np.deg2rad(rpy)

        Rm = _euler_zyx_to_R(rpy)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rm
        T[:3, 3] = t
        return T[:3, :4]  # مثل نسخه‌ی شما، 3×4 برمی‌گردانیم (در inverse هم پشتیبانی شده)

    # ------------------ نزدیکترین جواب ------------------

    def closest_solution(self, ee_pose, current_joint_angles=None, config_num=None):
        _, _, joint_configs = self.inverse(ee_pose)
        if joint_configs.size == 0:
            raise RuntimeError(f"No IK solution found for this end effector position {ee_pose}")

        if current_joint_angles is None:
            current_joint_angles = self.current_joint_angles

        if current_joint_angles is not None:
            diffs = joint_configs - current_joint_angles
            diffs = (diffs + np.pi) % (2.0*np.pi) - np.pi
            distances = np.linalg.norm(diffs, axis=1)
            config = int(np.argmin(distances))
        else:
            config = 0

        if config_num is not None:
            config = int(config_num)

        selected = joint_configs[config]
        if current_joint_angles is not None:
            selected = (selected - current_joint_angles + np.pi) % (2.0*np.pi) - np.pi + current_joint_angles

        self.current_joint_angles = selected
        return np.round(selected, 5)



if __name__ == '__main__':
    import os
    import time
    
    # os.system('cls')
    arm_kin = Kinematics('ur5e', file_path='./arms_config.yaml', optimized=True)
    thetas = [pi/2, -pi/3, pi/4, pi/10, pi/6, pi/20]
    
    n = 1000
    start_time = time.time()
    for i in range(n):
        ee_pose = arm_kin.forward(thetas)
        f, c, fc = arm_kin.inverse(ee_pose)
    end_time = time.time()
    print((end_time - start_time), 'avg:    ', (end_time - start_time)/n)
    
    print(ee_pose)

    print(thetas)
    print(f)
    print(fc)
