#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import rospy, moveit_commander
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from tf.transformations import (
    quaternion_from_euler, euler_from_quaternion,
    quaternion_matrix, quaternion_from_matrix, quaternion_multiply
)
from copy import deepcopy
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from UR_Kinematics import Kinematics

# ===================== Tunable Parameters =====================
DEFAULT_CFG_ID  = 3
SCRIPT1_VMAX    = 0.2   # m/s (Amin)
SCRIPT1_AMAX    = 0.6   # m/s^2 (for documentation)
SCRIPT1_NUM     = 100
WARMUP_SEC      = 0.3
REFERENCE_FRAME = "base_link"

# Fixed orientation (Note: these angles must correspond to the correct end-effector orientation in MoveIt / robot frame)
RPY_STD = [np.pi/2, 0.1, np.pi - 0.1]

# Printing-bed center (meters) — actual physical location in the robot base frame
# Use RViz / teach pendant to move the nozzle to your perceived "bed center + Z=0 reference height" and fill the readings here
BED_CENTER = np.array([0.5, 0.5, 0.0], dtype=float)

# ===== Unified expected joint order =====
JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

def _mat4(R3):
    M = np.eye(4)
    M[:3, :3] = R3
    return M

# ===================== Parse G-code (extract only XYZ, ignore E/F/Feed) =====================
def parse_xyz_from_gcode(gcode_path, bed_center=BED_CENTER):
    """
    Read .gcode, extract X/Y/Z in mm, convert to meters, and add bed_center offset
    in the robot coordinate frame.
    Ignore E/F and all other non-linear commands.
    """
    import re
    unit_mm = True     # default G21 (mm)
    abs_mode = True    # default G90
    cur = np.zeros(3, dtype=float)  # in meters
    pts = []
    pat = re.compile(r'([GMT]\d+)|([XYZ])([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
    with open(gcode_path, 'r') as f:
        for raw in f:
            line = raw.split(';',1)[0].strip()
            if not line:
                continue
            m = pat.findall(line)
            if not m:
                continue
            words = [(a if a else b, c) for a,b,c in m]
            cmd = next((w for w,_ in words if w and w[0] in ('G','M','T')), None)

            # Unit / mode
            if cmd == 'G21': unit_mm = True;  continue   # mm
            if cmd == 'G20': unit_mm = False; continue   # inch (not supported; treat numbers directly)
            if cmd == 'G90': abs_mode = True;  continue
            if cmd == 'G91': abs_mode = False; continue

            # Only process linear moves G0/G1
            if cmd not in ('G0','G1', None):
                continue

            touched = False
            for k,v in words:
                if k in ('X','Y','Z'):
                    val = float(v)
                    if unit_mm:
                        val *= 0.001  # mm → m
                    idx = 'XYZ'.index(k)
                    cur[idx] = val if abs_mode else cur[idx] + val
                    touched = True

            if touched:
                # Add bed center offset in robot frame
                pts.append(cur.copy() + np.asarray(bed_center, dtype=float))

    if len(pts) == 1:
        pts.append(pts[0].copy())
    return np.array(pts, dtype=float)

# ===================== XYZ path + fixed RPY → (t, q) =====================
def ik_path_from_xyz(way_xyz, rpy_std=RPY_STD, arm_kind="ur5e", cfg_num=DEFAULT_CFG_ID,
                     eef_step=0.005, v_max=SCRIPT1_VMAX):
    from math import ceil
    kin = Kinematics(arm_kind)

    # Densify polyline
    fine, last = [way_xyz[0]], way_xyz[0]
    for i in range(1, len(way_xyz)):
        p = way_xyz[i]
        L = float(np.linalg.norm(p-last))
        n = max(1, int(ceil(L/max(eef_step,1e-6))))
        for k in range(1, n+1):
            a = k/float(n)
            fine.append(last + a*(p-last))
        last = p
    fine = np.array(fine, dtype=float)

    # Time axis: arc-length uniform velocity (ignore F; use v_max)
    t = [0.0]
    for i in range(1, len(fine)):
        d = float(np.linalg.norm(fine[i]-fine[i-1]))
        t.append(t[-1] + d/max(v_max, 1e-4))
    t = np.array(t, dtype=float)

    # IK with fixed orientation
    Q, prev = [], None
    for p in fine:
        vec6 = np.concatenate([p, np.asarray(rpy_std, dtype=float)])
        ee = kin.make_ee_pose(vec6)
        try:
            q = kin.closest_solution(ee, current_joint_angles=prev, config_num=cfg_num)
            prev = q
        except RuntimeError:
            q = prev if prev is not None else np.zeros(6)
        Q.append(q)
    return t, np.array(Q, dtype=float)

# ===================== Energy Integrator =====================
class EnergyIntegrator:
    def __init__(self, joints_expected=6):
        self.joints_expected = joints_expected
        self.t0 = None
        self.prev_t = None
        self.ts, self.power_pos = [], []
        self.q_list, self.qd_list, self.tau_list = [], [], []
        self.names = None
        self.idxmap = None
        self.order  = list(JOINT_ORDER)
        self._enabled = False
        self.sub = rospy.Subscriber("/joint_states", JointState, self.cb, queue_size=200)

    def enable(self, on=True, reset=True):
        self._enabled = on
        if reset:
            self.t0 = self.prev_t = None
            self.ts[:] = []
            self.power_pos[:] = []
            self.q_list[:] = []
            self.qd_list[:] = []
            self.tau_list[:] = []
            self.names = None
            self.idxmap = None

    def cb(self, msg: JointState):
        if not self._enabled:
            return
        if self.idxmap is None:
            if not msg.name or len(msg.name) < self.joints_expected:
                return
            name_to_idx = {n: i for i, n in enumerate(list(msg.name))}
            if not all(n in name_to_idx for n in self.order):
                return
            self.idxmap = [name_to_idx[n] for n in self.order]
            self.names = list(self.order)
            rospy.loginfo("[EnergyIntegrator] /joint_states name order: %s", list(msg.name))
            rospy.loginfo("[EnergyIntegrator] Reordered to JOINT_ORDER: %s", self.names)

        t = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()
        q_all   = np.asarray(msg.position, dtype=float)
        qd_all  = np.asarray(msg.velocity, dtype=float)
        tau_all = np.asarray(msg.effort,   dtype=float)
        if q_all.size < max(self.idxmap)+1 or qd_all.size < max(self.idxmap)+1 or tau_all.size < max(self.idxmap)+1:
            return
        q   = q_all  [self.idxmap]
        qd  = qd_all [self.idxmap]
        tau = tau_all[self.idxmap]

        # Positive mechanical power only
        p_pos = float(np.sum(np.maximum(tau * qd, 0.0)))
        self.q_list.append(q)
        self.qd_list.append(qd)
        self.tau_list.append(tau)

        if self.prev_t is None:
            self.t0 = self.prev_t = t
            self.ts.append(0.0)
            self.power_pos.append(p_pos)
            return

        dt = t - self.prev_t
        if dt <= 0:
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

# ===================== Trajectory Builder =====================
def build_traj_from_t_q(group, t, q_ordered, warmup=WARMUP_SEC, dt_min=1e-3):
    active = group.get_active_joints()
    rospy.loginfo("[MoveIt] active joints: %s", active)
    name_to_order = {n: i for i, n in enumerate(JOINT_ORDER)}
    try:
        order_to_active_idx = [name_to_order[n] for n in active]
    except KeyError as e:
        raise RuntimeError(f"Active joint {e} not found in JOINT_ORDER {JOINT_ORDER}")
    qcur_active = np.array(group.get_current_joint_values(), dtype=float)

    traj = RobotTrajectory()
    traj.joint_trajectory.joint_names = active

    # Add warmup section
    times = [0.0, float(max(0.2, warmup))]
    for i, ti in enumerate(t):
        if i == 0 and float(ti) <= 0.0:
            continue
        times.append(float(warmup) + float(ti))
    for i in range(1, len(times)):
        if times[i] <= times[i - 1] + dt_min:
            times[i] = times[i - 1] + dt_min

    def ordered_to_active(q_ord):
        return np.asarray(q_ord, dtype=float)[order_to_active_idx]

    q_list_active = [qcur_active, ordered_to_active(q_ordered[0, :])]
    for i in range(1, len(t)):
        q_list_active.append(ordered_to_active(q_ordered[i, :]))

    pts = []
    for pos, tim in zip(q_list_active, times):
        pt = JointTrajectoryPoint()
        pt.positions = np.asarray(pos, dtype=float).tolist()
        pt.time_from_start = rospy.Duration(float(tim))
        pts.append(pt)

    # Simple velocity estimation
    for i in range(len(pts)):
        if i == 0 or i == len(pts) - 1:
            v = np.zeros_like(q_list_active[0])
        else:
            t0 = pts[i - 1].time_from_start.to_sec()
            t1 = pts[i + 1].time_from_start.to_sec()
            dt = max(t1 - t0, 1e-6)
            q0 = np.array(pts[i - 1].positions)
            q1 = np.array(pts[i + 1].positions)
            v  = (q1 - q0) / dt
        pts[i].velocities = v.tolist()

    assert np.all(np.diff([p.time_from_start.to_sec() for p in pts]) > 0.0), "non-increasing timestamps"
    traj.joint_trajectory.points = pts
    return traj

def execute_traj(group, traj):
    ok = False
    try:
        ok = group.execute(traj, wait=True)
    except Exception as e:
        rospy.logwarn("Execute exception: %s", e)
        ok = False
    group.stop()
    return bool(ok)

# ===================== FK for Visualization (Optional) =====================
def compute_ee_from_Q(ts, Q_ordered, joint_names_ordered, planning_frame, ee_link):
    try:
        from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
        from moveit_msgs.msg import RobotState
        rospy.wait_for_service('/compute_fk', timeout=3.0)
        fk = rospy.ServiceProxy('/compute_fk', GetPositionFK)
        positions = []
        for i in range(len(ts)):
            rs = RobotState()
            rs.joint_state.name = list(joint_names_ordered)
            rs.joint_state.position = Q_ordered[i, :].tolist()
            req = GetPositionFKRequest()
            req.header.frame_id = planning_frame
            req.fk_link_names = [ee_link]
            req.robot_state = rs
            res = fk(req)
            if res.error_code.val < 1 or not res.pose_stamped:
                positions.append(positions[-1] if positions else [np.nan] * 3)
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

# ===================== Trim warmup section =====================
def cut_after_boundary(ts, Q, ppos, QD, TAU, q_boundary):
    if len(ts) == 0 or len(Q) == 0:
        return ts, ppos, Q, QD, TAU
    qb = np.array(q_boundary, dtype=float).reshape(1, -1)
    d  = np.linalg.norm(Q - qb, axis=1)
    idx = int(np.argmin(d))
    idx = min(len(ts) - 1, max(1, idx + 1))
    t0  = ts[idx]
    ts2 = ts[idx:] - t0
    _c  = lambda a: a[idx:] if len(a) > idx else np.array([])
    return ts2, _c(ppos), _c(Q), _c(QD), _c(TAU)

# ===================== Main =====================
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter
    import argparse, rospkg

    rospy.init_node("ur5e_run_gcode_energy", anonymous=True)
    moveit_commander.roscpp_initialize([])

    group = moveit_commander.MoveGroupCommander("manipulator")
    group.set_pose_reference_frame(REFERENCE_FRAME)

    # Parse command line or use default G-code path
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcode', type=str, default='')
    parser.add_argument('--bed-center', type=float, nargs=3, metavar=('X','Y','Z'),
                        help='Override BED_CENTER in meters, robot base frame')
    args, _ = parser.parse_known_args()
    if args.gcode and os.path.isfile(args.gcode):
        gcode_path = args.gcode
    else:
        pkg = rospkg.RosPack().get_path('my_ur5_energy')
        gcode_path = os.path.join(pkg, 'scripts', 'Bunny_10.gcode')
    rospy.loginfo("Using G-code: %s", gcode_path)

    bed_center = BED_CENTER if args.bed_center is None else np.array(args.bed_center, dtype=float)

    # Read G-code → XYZ (meters, offset added; no axis flipping applied)
    way_xyz = parse_xyz_from_gcode(gcode_path, bed_center=bed_center)

    # Simple check: is Z stable within layer?
    if way_xyz.size:
        z = way_xyz[:,2]
        rospy.loginfo("G-code Z range: min=%.4f, max=%.4f, peak-to-peak=%.4f", z.min(), z.max(), z.max()-z.min())

    # XYZ + fixed RPY → (t, q)
    arm_kind = "ur5e"
    t_path, q_path_ordered = ik_path_from_xyz(
        way_xyz, rpy_std=RPY_STD, arm_kind=arm_kind,
        cfg_num=DEFAULT_CFG_ID, eef_step=0.005, v_max=SCRIPT1_VMAX
    )

    # Build and execute joint trajectory
    traj = build_traj_from_t_q(group, t_path, q_path_ordered, warmup=WARMUP_SEC, dt_min=1e-3)

    meter = EnergyIntegrator(joints_expected=6)
    meter.enable(on=True, reset=True)
    rospy.sleep(0.2)
    ok = execute_traj(group, traj)
    rospy.sleep(0.2)
    meter.stop()

    ts, ppos, Q_ordered, QD_ordered, TAU_ordered, names = meter.results()
    if len(ts) < 3 or len(ppos) != len(ts):
        rospy.logwarn("No data collected, skip plotting.")
        moveit_commander.roscpp_shutdown()
        return

    # Trim warmup
    ts, ppos, Q_ordered, QD_ordered, TAU_ordered = cut_after_boundary(
        ts, Q_ordered, ppos, QD_ordered, TAU_ordered, q_boundary=q_path_ordered[0]
    )

    # Compute energy
    E_pos = float(np.trapz(ppos, ts))
    E_signed = np.nan
    E_abs = np.nan
    E_pos_by_joint = E_neg_by_joint = None
    if len(ts) >= 2 and len(TAU_ordered) and len(QD_ordered):
        P_joint = TAU_ordered * QD_ordered
        P_pos   = np.maximum(P_joint, 0.0)
        P_neg   = np.minimum(P_joint, 0.0)
        E_signed = float(np.trapz(np.sum(P_joint, axis=1), ts))
        E_abs    = float(np.trapz(np.sum(np.abs(P_joint), axis=1), ts))
        E_pos_by_joint = np.trapz(P_pos, ts, axis=0)
        E_neg_by_joint = np.trapz(P_neg, ts, axis=0)
    dur = ts[-1] - ts[0] if len(ts) else 0.0
    P_avg_pos = (E_pos / dur) if dur > 0 else 0.0

    figdir = os.path.expanduser("~/ur_ws/src/my_ur5_energy/plots")
    os.makedirs(figdir, exist_ok=True)
    rospy.loginfo("Energy (G-code): E_pos=%.6f J | E_signed=%.6f J | E_abs=%.6f J | Duration=%.3f s | P_avg_pos=%.3f W",
                  E_pos, E_signed, E_abs, dur, P_avg_pos)
    out_path = os.path.join(figdir, "run_gcode_energy.txt")
    with open(out_path, "w") as f:
        f.write(f"E_pos_total (J)   = {E_pos:.6f}\n")
        f.write(f"E_signed_total (J)= {E_signed:.6f}\n")
        f.write(f"E_abs_total (J)   = {E_abs:.6f}\n")
        f.write(f"Duration (s)      = {dur:.6f}\n")
        f.write(f"P_avg_pos (W)     = {P_avg_pos:.6f}\n")
        if E_pos_by_joint is not None:
            f.write("E_pos_by_joint (J)= " + np.array2string(E_pos_by_joint, precision=6, separator=', ') + "\n")
            f.write("E_neg_by_joint (J)= " + np.array2string(E_neg_by_joint, precision=6, separator=', ') + "\n")
    rospy.loginfo("Saved energy summary to: %s", out_path)

    # Plotting (same as original, slightly organized)
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter
    dt_med = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.01
    win = max(5, int(round(0.20 / max(dt_med, 1e-3))) | 1)
    try:
        ppos_smooth = savgol_filter(ppos, win, 3) if len(ppos) >= win else ppos * 1.0
    except Exception:
        ppos_smooth = ppos * 1.0

    try:
        if len(Q_ordered) and len(ts) > 1 and Q_ordered.ndim == 2:
            QD_plot  = savgol_filter(Q_ordered,  win, 3, deriv=1, delta=dt_med, axis=0)
            QDD_plot = savgol_filter(Q_ordered,  win, 3, deriv=2, delta=dt_med, axis=0)
        else:
            raise RuntimeError("not enough data for savgol on Q")
    except Exception:
        if len(QD_ordered) and len(ts) > 1:
            QD_plot  = QD_ordered.copy()
            QDD_plot = np.gradient(QD_plot, ts, axis=0)
        else:
            QD_plot  = Q_ordered if isinstance(Q_ordered, np.ndarray) else np.zeros((len(ts), 0))
            QDD_plot = np.zeros_like(QD_plot)

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (4.2, 3.0),
        "figure.dpi": 200,
        "lines.linewidth": 1.8,
        "axes.grid": False,
        "grid.color": "#B0B0B0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.6,
    })

    X_LIMIT       = None
    Y_EXPAND      = 1.5
    LEGEND_FONT   = 8
    LEGEND_SHRINK = 0.85
    LEGEND_LOC    = "upper right"
    LEGEND_ANCHOR = (0.98, 0.98)

    def _styled_plot(x, Y, y_label, series_labels,
                     xlim=X_LIMIT, y_expand=Y_EXPAND,
                     legend_loc=LEGEND_LOC, legend_anchor=LEGEND_ANCHOR,
                     legend_fontsize=LEGEND_FONT, legend_shrink=LEGEND_SHRINK,
                     filepath=None):
        fig, ax = plt.subplots()
        Y = np.atleast_2d(Y)
        for i, y in enumerate(Y.T):
            ls = ["-", "--", "-.", ":"][i % 4]
            ax.plot(x, y, linestyle=ls, label=series_labels[i])
        ax.set_xlabel("time (s)"); ax.set_ylabel(y_label)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        else:
            x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
            ax.set_xlim(x_min, x_max + 0.10 * (x_max - x_min))
        if Y.size:
            y_min, y_max = float(np.nanmin(Y)), float(np.nanmax(Y))
            y_center = 0.5 * (y_max + y_min)
            half = max(abs(y_max - y_center), abs(y_center - y_min))
            half = max(half, 1e-6) * float(y_expand)
            ax.set_ylim(y_center - half, y_center + half)
        ax.minorticks_on()
        ax.set_axisbelow(True)
        ax.grid(True, which="major", color="#B0B0B0", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(True, which="minor", color="#E0E0E0", linestyle="-", linewidth=0.5, alpha=0.35)
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_anchor, ncol=1, fontsize=legend_fontsize,
                  frameon=True, framealpha=0.95, fancybox=True,
                  handlelength=1.5 * legend_shrink, handletextpad=0.4 * legend_shrink,
                  labelspacing=0.25 * legend_shrink, borderpad=0.25 * legend_shrink, borderaxespad=0.2)
        fig.tight_layout()
        if filepath:
            fig.savefig(filepath, dpi=300, bbox_inches="tight"); plt.close(fig)
        else:
            plt.show()
        return fig, ax

    def _labels(prefix, arr):
        n = int(arr.shape[1]) if isinstance(arr, np.ndarray) and arr.ndim == 2 else 0
        return [f"{prefix}{i+1}" for i in range(n)]

    q_labels   = _labels("q",   Q_ordered)
    qd_labels  = _labels("dq",  QD_plot)
    qdd_labels = _labels("ddq", QDD_plot)
    tau_labels = _labels("tau", TAU_ordered)
    xyz_labels = ["x", "y", "z"]

    if len(ppos):
        _styled_plot(ts, ppos.reshape(-1, 1), "Power (W)", ["power"],
                     filepath=os.path.join(figdir, "run_gcode_power.png"))
    if len(Q_ordered):
        _styled_plot(ts, Q_ordered, "Joint position (rad)", q_labels,
                     filepath=os.path.join(figdir, "run_gcode_q.png"))
    if 'QD_plot' in locals() and len(QD_plot):
        _styled_plot(ts, QD_plot, "Joint velocity (rad/s)", qd_labels,
                     filepath=os.path.join(figdir, "run_gcode_qd.png"))
    if 'QDD_plot' in locals() and len(QDD_plot):
        _styled_plot(ts, QDD_plot, "Joint acceleration (rad/s^2)", qdd_labels,
                     filepath=os.path.join(figdir, "run_gcode_qdd.png"))
    if len(TAU_ordered):
        _styled_plot(ts, TAU_ordered, "Torque (N*m)", tau_labels,
                     filepath=os.path.join(figdir, "run_gcode_tau.png"))

    planning_frame = group.get_planning_frame()
    ee_link = group.get_end_effector_link() or group.get_link_names()[-1]
    if len(ts) and len(Q_ordered):
        P, dP, speed, fk_err = compute_ee_from_Q(ts, Q_ordered, JOINT_ORDER, planning_frame, ee_link)
        if P is not None and np.isfinite(P).all():
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(P[:, 0], P[:, 1], P[:, 2], linestyle="-")
            ax.scatter(P[0, 0],  P[0, 1],  P[0, 2],  marker="o")
            ax.scatter(P[-1, 0], P[-1, 1], P[-1, 2], marker="x")
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
            fig.tight_layout()
            fig.savefig(os.path.join(figdir, "run_gcode_ee_traj3d.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
            _styled_plot(ts, P,  "Position (m)", xyz_labels,
                         filepath=os.path.join(figdir, "run_gcode_ee_pos.png"))
            _styled_plot(ts, np.linalg.norm(P, axis=1).reshape(-1, 1),
                         "Position magnitude (m)", ["pos_mag"],
                         filepath=os.path.join(figdir, "run_gcode_ee_pos_mag.png"))
            if dP is not None:
                _styled_plot(ts, dP, "Linear velocity (m/s)", xyz_labels,
                             filepath=os.path.join(figdir, "run_gcode_ee_vel.png"))
                try:
                    ddP = np.gradient(dP, ts, axis=0)
                except Exception:
                    ddP = None
                if ddP is not None and np.isfinite(ddP).all():
                    _styled_plot(ts, ddP, "Linear acceleration (m/s^2)",
                                 ["ax", "ay", "az"],
                                 filepath=os.path.join(figdir, "run_gcode_ee_acc.png"))
                    _styled_plot(ts, np.linalg.norm(ddP, axis=1).reshape(-1, 1),
                                 "Acceleration magnitude (m/s^2)", ["acc"],
                                 filepath=os.path.join(figdir, "run_gcode_ee_acc_mag.png"))
            if speed is not None:
                _styled_plot(ts, speed.reshape(-1, 1), "Speed magnitude (m/s)", ["speed"],
                             filepath=os.path.join(figdir, "run_gcode_ee_speed.png"))

    rospy.loginfo("Saved plots to: %s", figdir)
    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    main()
