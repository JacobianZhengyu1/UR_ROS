#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import rospy, moveit_commander
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
from copy import deepcopy
from moveit_msgs.msg import RobotTrajectory

# ===================== 你可以修改的默认参数（直接改这里即可） =====================
# 关节初始解种子（用来帮助P1规划通过）
DEFAULT_CFG_ID     = 3
# MoveIt 缩放
DEFAULT_VEL_SCALE  = 1
DEFAULT_ACC_SCALE  = 1
# 速度/加速度上限（再二次限幅）
DEFAULT_VMAX       = 1
DEFAULT_AMAX       = 1
# 直线插值步长（米）
DEFAULT_EEF_STEP   = 0.02
# 规划时间与尝试次数
DEFAULT_PLANNING_TIME = 20.0
DEFAULT_ATTEMPTS      = 30
# 坐标点（已写死在脚本里）
# 注意：将 z 提高到 0.30，避免碰地
P1_XYZ = [-0.5,  0.1, 0]          # 到达 P1：允许任意路径（非直线）
P2_XYZ = [-0.3, -0.1, 0.3]          # 从 P1 到 P2：要求笛卡尔直线
# 姿态（RPY，弧度）
RPY    = [np.pi - 0.1, 0.1, np.pi/2]
# 直线段是否保持姿态与 P1 一致（True=保持；False=允许使用 P2 的姿态）
KEEP_RPY_ON_LINE = True
# 参考坐标系
REFERENCE_FRAME = "base_link"
# ============================================================================

# ===================== 功率/状态采集器 =====================
class EnergyIntegrator:
    """只记录正功功率 power_pos = sum_i max(tau_i*qdot_i,0) 与 q/qd/tau/时间戳"""
    def __init__(self, joints_expected=6):
        self.joints_expected = joints_expected
        self.t0 = None
        self.prev_t = None
        self.ts = []
        self.power_pos = []
        self.q_list, self.qd_list, self.tau_list = [], [], []
        self.names = None
        self._enabled = False
        self.sub = rospy.Subscriber("/joint_states", JointState, self.cb, queue_size=200)

    def enable(self, on=True, reset=True):
        self._enabled = on
        if reset:
            self.t0 = self.prev_t = None
            self.ts[:] = []; self.power_pos[:] = []
            self.q_list[:] = []; self.qd_list[:] = []; self.tau_list[:] = []
            self.names = None

    def cb(self, msg: JointState):
        if not self._enabled:
            return
        t = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()
        if self.names is None and msg.name:
            self.names = list(msg.name[:self.joints_expected])

        q   = np.array(msg.position[:self.joints_expected], dtype=float)
        qd  = np.array(msg.velocity[:self.joints_expected], dtype=float)
        tau = np.array(msg.effort  [:self.joints_expected], dtype=float)
        p_pos = float(np.sum(np.maximum(tau*qd, 0.0)))

        self.q_list.append(q); self.qd_list.append(qd); self.tau_list.append(tau)

        if self.prev_t is None:
            self.t0 = self.prev_t = t
            self.ts.append(0.0)
            self.power_pos.append(p_pos)
            return

        dt = t - self.prev_t
        if dt <= 0:  # 丢弃倒序/重复时间戳
            return

        self.ts.append(t - self.t0)
        self.power_pos.append(p_pos)
        self.prev_t = t

    def stop(self):
        self.sub.unregister()

    def results(self):
        return (np.array(self.ts),
                np.array(self.power_pos),
                np.array(self.q_list),
                np.array(self.qd_list),
                np.array(self.tau_list),
                self.names)

# ===================== MoveIt 工具 =====================
def ensure_exec_limits(group, vel_scale, acc_scale):
    group.set_max_velocity_scaling_factor(max(0.01, min(vel_scale, 1.0)))
    group.set_max_acceleration_scaling_factor(max(0.01, min(acc_scale, 1.0)))

CFG_SEEDS = {
    0: [ 0.0, -1.57,  1.57, 0.0,  1.57, 0.0],
    1: [ 0.0, -1.57,  1.57, 0.0, -1.57, 0.0],
    2: [ 0.0, -1.20,  1.20, 0.0,  1.57, 0.0],
    3: [ 0.0, -1.20,  1.20, 0.0, -1.57, 0.0],
    4: [ 0.0, -2.10,  2.10, 0.0,  1.57, 0.0],
    5: [ 0.0, -2.10,  2.10, 0.0, -1.57, 0.0],
    6: [ 0.5, -1.57,  1.57, 0.0,  1.57, 0.0],
    7: [-0.5, -1.57,  1.57, 0.0, -1.57, 0.0],
}

def set_seed_to_config(group, cfg_id):
    try:
        from moveit_msgs.msg import RobotState
        qnames = group.get_active_joints()
        seed = CFG_SEEDS.get(int(cfg_id))
        if seed is None or len(seed) != len(qnames):
            return
        rs = RobotState()
        rs.joint_state.name = qnames
        rs.joint_state.position = seed
        group.set_start_state(rs)
    except Exception:
        pass

def set_pose_xyz_rpy(pose: Pose, x, y, z, r, p, yaw):
    qx, qy, qz, qw = quaternion_from_euler(r, p, yaw)
    pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z)
    pose.orientation.x, pose.orientation.y = qx, qy
    pose.orientation.z, pose.orientation.w = qz, qw
    return pose

def _as_traj(plan):
    if plan is None:
        return None
    if hasattr(plan, "joint_trajectory"):
        t = RobotTrajectory()
        t.joint_trajectory = plan.joint_trajectory
        return t
    if isinstance(plan, tuple) and len(plan) >= 1:
        c0 = plan[0]
        if hasattr(c0, "joint_trajectory"):
            t = RobotTrajectory()
            t.joint_trajectory = c0.joint_trajectory
            return t
        if len(plan) > 1 and hasattr(plan[1], "joint_trajectory"):
            t = RobotTrajectory()
            t.joint_trajectory = plan[1].joint_trajectory
            return t
    return None

def plan_to_point(group, frame, x, y, z, rpy=None, planning_time=20.0, attempts=30):
    current = group.get_current_pose().pose
    if rpy is None:
        target = Pose()
        target.position.x, target.position.y, target.position.z = x, y, z
        target.orientation = current.orientation
    else:
        r, p, yaw = rpy
        target = set_pose_xyz_rpy(Pose(), x, y, z, r, p, yaw)

    group.set_pose_reference_frame(frame)
    group.set_start_state_to_current_state()
    group.set_planning_time(planning_time)
    group.set_num_planning_attempts(attempts)
    group.set_pose_target(target)
    plan_raw = group.plan()
    traj = _as_traj(plan_raw)
    group.clear_pose_targets()
    return traj

def plan_cartesian_line_from_state(group, frame, start_state_rs, p2_xyz, rpy_end, eef_step=0.01):
    """从给定 start_state 开始，直线到 P2；waypoints 只放 P2。兼容多种 MoveIt 绑定签名。"""
    group.set_pose_reference_frame(frame)
    group.set_start_state(start_state_rs)

    waypoints = []
    tp = set_pose_xyz_rpy(Pose(), p2_xyz[0], p2_xyz[1], p2_xyz[2], *rpy_end)
    waypoints.append(tp)

    # 兼容性调用顺序：
    # S1: (wps, eef_step, avoid_collisions:bool)
    # S2: (wps, eef_step, avoid_collisions:bool, path_constraints:ByteString)
    # S3: (wps, eef_step, jump_threshold:float, avoid_collisions:bool)
    try:
        plan, fraction = group.compute_cartesian_path(waypoints, float(eef_step), True)
    except Exception:
        try:
            plan, fraction = group.compute_cartesian_path(waypoints, float(eef_step), True, b'')
        except Exception:
            plan, fraction = group.compute_cartesian_path(waypoints, float(eef_step), 0.0, True)
    return plan, float(fraction)



def execute_traj(group, traj):
    ok = False
    try:
        ok = group.execute(traj, wait=True)
    except Exception as e:
        rospy.logwarn("Execute exception: %s", e)
        ok = False
    group.stop()
    return bool(ok)

def retime_and_cap(group, traj, vmax=None, amax=None, vel_scale=1.0, acc_scale=1.0):
    new_traj = group.retime_trajectory(
        group.get_current_state(), traj,
        velocity_scaling_factor=max(0.01, min(vel_scale, 1.0)),
        acceleration_scaling_factor=max(0.01, min(acc_scale, 1.0))
    )
    jt = new_traj.joint_trajectory
    if (vmax is None and amax is None) or not jt.points:
        return new_traj

    for i in range(1, len(jt.points)):
        p0, p1 = jt.points[i-1], jt.points[i]
        t0 = p0.time_from_start.to_sec()
        dt = max(p1.time_from_start.to_sec() - t0, 1e-6)

        s_v = 1.0
        if vmax is not None and getattr(p1, "velocities", []):
            s_v = max([1.0] + [abs(v)/float(vmax) for v in p1.velocities])

        s_a = 1.0
        if amax is not None and getattr(p1, "accelerations", []):
            s_a = max([1.0] + [abs(a)/float(amax) for a in p1.accelerations])

        s = max(s_v, s_a)
        if s > 1.0:
            dt *= s
        p1.time_from_start = rospy.Duration(t0 + dt)
    return new_traj

def concat_trajectories(traj_a, traj_b):
    out = RobotTrajectory()
    out.joint_trajectory.joint_names = traj_a.joint_trajectory.joint_names
    pts = []
    pts.extend(traj_a.joint_trajectory.points)
    t_offset = pts[-1].time_from_start
    for p in traj_b.joint_trajectory.points:
        q = deepcopy(p)
        q.time_from_start = t_offset + q.time_from_start
        pts.append(q)
    out.joint_trajectory.points = pts
    return out

# ---- FK：从关节角计算末端位置
def compute_ee_from_Q(ts, Q, joint_names, planning_frame, ee_link):
    try:
        from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
        from moveit_msgs.msg import RobotState
        rospy.wait_for_service('/compute_fk', timeout=3.0)
        fk = rospy.ServiceProxy('/compute_fk', GetPositionFK)

        positions = []
        for i in range(len(ts)):
            rs = RobotState()
            rs.joint_state.name = joint_names
            rs.joint_state.position = Q[i,:].tolist()
            req = GetPositionFKRequest()
            req.header.frame_id = planning_frame
            req.fk_link_names = [ee_link]
            req.robot_state = rs
            res = fk(req)
            if res.error_code.val < 1 or not res.pose_stamped:
                positions.append(positions[-1] if positions else [np.nan]*3)
            else:
                p = res.pose_stamped[0].pose.position
                positions.append([p.x, p.y, p.z])
        P = np.array(positions, dtype=float)

        if len(ts) > 1:
            dp = np.gradient(P, ts, axis=0)
            speed = np.linalg.norm(dp, axis=1)
        else:
            dp = np.zeros_like(P)
            speed = np.zeros((len(ts),))
        return P, dp, speed, None
    except Exception as e:
        rospy.logwarn("FK unavailable or failed, skip EE plots: %s", e)
        return None, None, None, e

# ---- 用“实测关节角”定位并裁切第一段之后的数据
def cut_after_measured_boundary(ts, Q, ppos, QD, TAU, q_boundary, T1_hint=None, guard=1.0):
    if len(ts) == 0 or len(Q) == 0:
        return ts, ppos, Q, QD, TAU

    q_boundary = np.array(q_boundary, dtype=float).reshape(1, -1)
    d = np.linalg.norm(Q - q_boundary, axis=1)

    if T1_hint is not None and np.isfinite(T1_hint):
        mask = (ts >= (T1_hint - guard)) & (ts <= (T1_hint + guard))
        if mask.any():
            idx0 = np.nonzero(mask)[0][0]
            idx  = idx0 + int(np.argmin(d[mask]))
        else:
            idx = int(np.argmin(np.abs(ts - T1_hint)))
    else:
        idx = int(np.argmin(d))

    idx = min(len(ts)-1, max(1, idx+1))  # 从该点之后开始
    t0  = ts[idx]
    ts2 = ts[idx:] - t0

    def _cut(a): return a[idx:] if len(a) > idx else np.array([])
    return ts2, _cut(ppos), _cut(Q), _cut(QD), _cut(TAU)

# ===================== 单次运行：到 P1 + 直线到 P2 =====================
def run_once(group, frame, p1_xyz, rpy1, p2_xyz, rpy2, keep_rpy,
             vel=DEFAULT_VEL_SCALE, acc=DEFAULT_ACC_SCALE,
             vmax=DEFAULT_VMAX, amax=DEFAULT_AMAX,
             planning_time=DEFAULT_PLANNING_TIME, attempts=DEFAULT_ATTEMPTS,
             eef_step=DEFAULT_EEF_STEP, arm_config=DEFAULT_CFG_ID):

    from moveit_msgs.msg import RobotState
    set_seed_to_config(group, arm_config)

    # 1) 到 P1（自由规划）
    traj1 = plan_to_point(group, frame, *p1_xyz, rpy=rpy1,
                          planning_time=planning_time, attempts=attempts)
    if not (traj1 and len(traj1.joint_trajectory.points) > 0):
        rospy.logwarn("Failed to plan to P1.")
        return (tuple([np.array([]) for _ in range(6)])), False
    traj1r = retime_and_cap(group, traj1, vmax=vmax, amax=amax, vel_scale=vel, acc_scale=acc)

    # 构造“到 P1 之后”的 start_state
    q_end1 = np.array(traj1r.joint_trajectory.points[-1].positions, dtype=float)
    rs_end1 = RobotState()
    rs_end1.joint_state.name = group.get_active_joints()
    rs_end1.joint_state.position = q_end1.tolist()

    # 2) 直线到 P2
    if keep_rpy:
        rpy_end = rpy1                     # 保持姿态与 P1 相同
    else:
        rpy_end = rpy2 if rpy2 is not None else rpy1

    traj2, frac = plan_cartesian_line_from_state(group, frame, rs_end1, p2_xyz, rpy_end, eef_step=eef_step)
    if not (traj2 and len(traj2.joint_trajectory.points) > 0):
        rospy.logwarn("Failed to plan Cartesian to P2.")
        return (tuple([np.array([]) for _ in range(6)])), False
    rospy.loginfo("Cartesian path fraction = %.2f", frac)
    traj2r = retime_and_cap(group, traj2, vmax=vmax, amax=amax, vel_scale=vel, acc_scale=acc)

    # 记录直线段起止关节
    q_start_cart = q_end1.copy()
    q_end_cart   = np.array(traj2r.joint_trajectory.points[-1].positions, dtype=float)
    joint_order  = group.get_active_joints()
    rospy.loginfo("Joint order: %s", str(joint_order))
    rospy.loginfo("Joint positions (start of Cartesian) = %s", np.array2string(q_start_cart, precision=6, separator=', '))
    rospy.loginfo("Joint positions (end of Cartesian)   = %s", np.array2string(q_end_cart,   precision=6, separator=', '))

    # 保存到文件
    try:
        dump_dir = os.path.expanduser("~/ur_ws/src/my_ur5_energy/plots")
        os.makedirs(dump_dir, exist_ok=True)
        dump_path = os.path.join(dump_dir, "joints_start_end.txt")
        with open(dump_path, "w") as f:
            f.write("joint_order = {}\n".format(joint_order))
            f.write("start(rad) = {}\n".format(np.array2string(q_start_cart, precision=6, separator=', ')))
            f.write("end(rad)   = {}\n".format(np.array2string(q_end_cart,   precision=6, separator=', ')))
        rospy.loginfo("Saved start/end joints to: %s", dump_path)
    except Exception as e:
        rospy.logwarn("Failed to save joints file: %s", e)

    # 3) 合并并统一时间参数化
    comb = concat_trajectories(traj1r, traj2r)
    combr = retime_and_cap(group, comb, vmax=vmax, amax=amax, vel_scale=vel, acc_scale=acc)

    # 第一段结束时间与关节向量（用于切图）
    n1 = len(traj1r.joint_trajectory.points)
    T1 = combr.joint_trajectory.points[n1-1].time_from_start.to_sec()
    q_boundary = np.array(combr.joint_trajectory.points[n1-1].positions, dtype=float)

    # 采集功率
    meter = EnergyIntegrator(joints_expected=6)
    meter.enable(on=True, reset=True); rospy.sleep(0.2)
    ok = execute_traj(group, combr)
    rospy.sleep(0.2); meter.stop()

    ts, ppos, Q, QD, TAU, names = meter.results()
    # 只保留第二段 P1->P2
    ts2, ppos2, Q2, QD2, TAU2 = cut_after_measured_boundary(
        ts, Q, ppos, QD, TAU, q_boundary=q_boundary, T1_hint=T1, guard=1.0
    )
    return (ts2, ppos2, Q2, QD2, TAU2, names), ok

# ===================== 主流程（生成图） =====================
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    rospy.init_node("ur5_line_energy", anonymous=True)
    moveit_commander.roscpp_initialize([])

    group = moveit_commander.MoveGroupCommander("manipulator")
    group.set_goal_position_tolerance(0.005)
    group.set_goal_orientation_tolerance(0.01)
    ensure_exec_limits(group, DEFAULT_VEL_SCALE, DEFAULT_ACC_SCALE)

    figdir = os.path.expanduser("~/ur_ws/src/my_ur5_energy/plots")
    os.makedirs(figdir, exist_ok=True)

    p1_xyz = list(P1_XYZ)
    p2_xyz = list(P2_XYZ)
    rpy1   = list(RPY)
    rpy2   = list(RPY)  # 如需在直线段更改姿态，可单独改这个

    rospy.loginfo("Run cfg=%s | keep_rpy=%s | p1=%s rpy1=%s | p2=%s rpy2=%s",
                  str(DEFAULT_CFG_ID), str(KEEP_RPY_ON_LINE),
                  np.array2string(np.array(p1_xyz), precision=3),
                  np.array2string(np.array(rpy1),   precision=3),
                  np.array2string(np.array(p2_xyz), precision=3),
                  np.array2string(np.array(rpy2),   precision=3))

    (ts, ppos, Q, QD, TAU, names), ok = run_once(
        group, REFERENCE_FRAME, p1_xyz, rpy1, p2_xyz, rpy2, KEEP_RPY_ON_LINE,
        vel=DEFAULT_VEL_SCALE, acc=DEFAULT_ACC_SCALE,
        vmax=DEFAULT_VMAX, amax=DEFAULT_AMAX,
        planning_time=DEFAULT_PLANNING_TIME, attempts=DEFAULT_ATTEMPTS,
        eef_step=DEFAULT_EEF_STEP, arm_config=DEFAULT_CFG_ID
    )

    if len(ts) < 3 or len(ppos) != len(ts):
        rospy.logwarn("No data collected, skip plotting.")
        moveit_commander.roscpp_shutdown()
        return

    # ========= 平滑参数（仅用于展示） =========
    dt_med = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.01
    win = max(5, int(round(max(0.20, 0.05)/max(dt_med, 1e-3))) | 1)  # 0.20s窗口，奇数
    try:
        ppos_smooth = savgol_filter(ppos, win, 3)
        QD_plot  = savgol_filter(Q,  win, 3, deriv=1, delta=dt_med, axis=0)
        QDD_plot = savgol_filter(Q,  win, 3, deriv=2, delta=dt_med, axis=0)
    except Exception:
        ppos_smooth = ppos*1.0
        QD_plot, QDD_plot = QD, np.gradient(QD, ts, axis=0) if len(ts) > 1 else QD*0.0

    # ========= 1) Power（只画正功） =========
    plt.figure(figsize=(6.0,4.2), dpi=120)
    plt.title("Power vs time")
    plt.plot(ts, ppos_smooth)
    plt.xlabel("time (s)"); plt.ylabel("Power (W)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "runA_power.png"), dpi=150, bbox_inches="tight")

    # ========= 2) joints position =========
    plt.figure(); plt.title("joints position vs time")
    if len(Q): plt.plot(ts, Q)
    plt.xlabel("time (s)"); plt.ylabel("q (rad)"); plt.grid(True)
    plt.savefig(os.path.join(figdir, "runA_q.png"), dpi=150, bbox_inches="tight")

    # ========= 3) joints velocity =========
    plt.figure(); plt.title("joints velocities vs time")
    if len(QD_plot): plt.plot(ts, QD_plot)
    plt.xlabel("time (s)"); plt.ylabel("qdot (rad/s)"); plt.grid(True)
    plt.savefig(os.path.join(figdir, "runA_qd.png"), dpi=150, bbox_inches="tight")

    # ========= 4) joints acceleration =========
    plt.figure(); plt.title("joints acceleration vs time")
    if len(QDD_plot): plt.plot(ts, QDD_plot)
    plt.xlabel("time (s)"); plt.ylabel("qddot (rad/s^2)"); plt.grid(True)
    plt.savefig(os.path.join(figdir, "runA_qdd.png"), dpi=150, bbox_inches="tight")

    # ========= 5) joints torques =========
    plt.figure(); plt.title("joints torques vs time (logged)")
    if len(TAU): plt.plot(ts, TAU)
    plt.xlabel("time (s)"); plt.ylabel("tau (Nm)"); plt.grid(True)
    plt.savefig(os.path.join(figdir, "runA_tau.png"), dpi=150, bbox_inches="tight")

    # ========= 6-9) 末端轨迹/速度 =========
    planning_frame = group.get_planning_frame()
    ee_link = group.get_end_effector_link() or group.get_link_names()[-1]
    if len(ts) and len(Q) and names is not None:
        P, dP, speed, fk_err = compute_ee_from_Q(ts, Q, names, planning_frame, ee_link)
        if P is not None and np.isfinite(P).all():
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
            ax.set_title("EE position trajectory")
            ax.plot(P[:,0], P[:,1], P[:,2]); ax.scatter(P[0,0], P[0,1], P[0,2], marker='o')
            ax.scatter(P[-1,0], P[-1,1], P[-1,2], marker='x')
            plt.tight_layout()
            plt.savefig(os.path.join(figdir, "runA_ee_traj3d.png"), dpi=150, bbox_inches="tight")

            plt.figure(); plt.title("EE position vs time")
            plt.plot(ts, P[:,0], label="x"); plt.plot(ts, P[:,1], label="y"); plt.plot(ts, P[:,2], label="z")
            plt.xlabel("time (s)"); plt.ylabel("pos (m)"); plt.grid(True); plt.legend()
            plt.savefig(os.path.join(figdir, "runA_ee_pos.png"), dpi=150, bbox_inches="tight")

            if dP is not None:
                plt.figure(); plt.title("EE velocity vs time")
                plt.plot(ts, dP[:,0], label="vx"); plt.plot(ts, dP[:,1], label="vy"); plt.plot(ts, dP[:,2], label="vz")
                plt.xlabel("time (s)"); plt.ylabel("vel (m/s)"); plt.grid(True); plt.legend()
                plt.savefig(os.path.join(figdir, "runA_ee_vel.png"), dpi=150, bbox_inches="tight")

            if speed is not None:
                plt.figure(); plt.title("EE speed magnitude vs time")
                plt.plot(ts, speed); plt.xlabel("time (s)"); plt.ylabel("|v| (m/s)"); plt.grid(True)
                plt.savefig(os.path.join(figdir, "runA_ee_speed.png"), dpi=150, bbox_inches="tight")
        else:
            rospy.logwarn("Skip EE plots (FK unavailable or invalid).")

    rospy.loginfo("Saved plots to: %s", figdir)
    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    main()
