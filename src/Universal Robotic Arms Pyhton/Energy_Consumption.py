import numpy as np
from collections import defaultdict
from functools import partial

from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

from _UR_Kinematics import Kinematics
from _UR_Dynamics import Dynamics

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import time


class Arm:
    def __init__(self, arm_type="ur5e"):

        defined_arms = ["ur5", "ur5e"]
        if arm_type not in defined_arms:
            raise NameError(
                f"The arm type of {arm_type} is not defined! Try one on these: {defined_arms}"
            )

        self.arm_kin = Kinematics(arm_type)
        self.arm_dyn = Dynamics(arm_type)

        self.arm_num_joints = 6  # change it later base on the selected 'arm_type' when loading kinematics or dynamics
        self.max_num_arm_configs = 8  # change it later base on the selected 'arm_type' when loading kinematics or dynamics
        self.num_freedom = 6
        self.motion_modes = ["ptp", "jtj", "ctc"]
        self.motion_profiles = ["trapezoidal", "cycloidal"]

    def assign_and_check_keywords(self, check_keyword=True, **kwargs):

        self.motion_mode = kwargs.get("motion_mode")
        self.motion_profile = kwargs.get("profile", "trapezoidal")

        if check_keyword:
            if self.motion_profile not in self.motion_profiles:
                raise NameError(
                    f"The profile type of {self.motion_profile} is not defined! Try one on these: {self.motion_profiles}"
                )

            if self.motion_mode not in self.motion_modes:
                raise NameError(
                    f"The motion mode type of {self.motion_mode} is not defined! Try one on these: {self.motion_modes}"
                )

        self.sync = kwargs.get("sync", True)
        self.num = kwargs.get("num", 20)
        self.window_length = kwargs.get("window_size", 10)
        self.poly_order = kwargs.get("poly_order", 3)
        self.singularities_limit = kwargs.get("singularities_limit", 10)

        self.arm_config = kwargs.get("arm_config", None)
        if check_keyword:
            if (
                self.arm_config is not None
            ) and self.arm_config >= self.max_num_arm_configs:
                raise ValueError(
                    f" arm_config must be between 0 to {self.max_num_arm_configs - 1}"
                )

        if self.motion_mode == "jtj":
            self.num_joints = kwargs.get("num_joints", 6)
            self.num_freedom = 6
        elif self.motion_mode == "ptp":
            self.num_joints = self.arm_num_joints
            self.num_freedom = kwargs.get("num_freedom", 6)
        elif self.motion_mode == "ctc":
            self.num_joints = self.arm_num_joints
            self.num_freedom = kwargs.get("num_freedom", 6)

        int_parameters = ["num", "num_freedom", "num_joints"]
        if check_keyword:
            for parameter in int_parameters:
                if parameter in kwargs and not isinstance(kwargs[parameter], int):
                    raise TypeError(f"Parameter {parameter} should be astype of int!")

        self.start_vector = np.array(kwargs.get("start_vector", None)).reshape(-1)
        self.end_vector = np.array(kwargs.get("end_vector", None)).reshape(-1)

        # check the size of start and end vectors to match with num_joint or num_freedom base on motion_mode
        vectores = ["start_vector", "end_vector"]
        if check_keyword:
            if self.motion_mode == "jtj":
                vector_length_limit = self.num_joints
                for vector in vectores:
                    if (
                        not len(np.array(kwargs[vector]).reshape(-1))
                        == vector_length_limit
                    ):
                        raise ValueError(
                            f"The number of elements in '{vector}' does not match with number of joints ({vector_length_limit})"
                        )

            elif self.motion_mode == "ptp":
                vector_length_limit = self.num_freedom
                for vector in vectores:
                    if (
                        not len(np.array(kwargs[vector]).reshape(-1))
                        == vector_length_limit
                    ):
                        raise ValueError(
                            f"The length of '{vector}' does not match with degrees of freedom ({vector_length_limit})"
                        )

        self.v_max = kwargs.get("v_max", 0.1)
        self.a_max = kwargs.get("a_max", 0.1)
        self.w_max = abs(kwargs.get("w_max", 0.2))
        self.alpha_max = abs(kwargs.get("alpha_max", 0.2))

        if check_keyword:
            if len(np.array(self.v_max).reshape(-1)) != 1:
                raise ValueError(f"The number of elements in 'v_max' should be 1")
            if len(np.array(self.a_max).reshape(-1)) != 1:
                raise ValueError(f"The number of elements in 'a_max' should be 1")

        self.vector_v_max = kwargs.get("vector_v_max")
        self.vector_a_max = kwargs.get("vector_a_max")

        if check_keyword:
            if "v_max" in kwargs and "vector_v_max" in kwargs:
                raise ValueError(
                    "Both v_max and vector_v_max got defined, Don't Know which one to use!"
                )

            if "a_max" in kwargs and "vector_a_max" in kwargs:
                raise ValueError(
                    "Both v_max and vector_v_max got defined, Don't Know which one to use!"
                )

        if check_keyword:
            if "v_max" in kwargs:
                if "vector_v_max" not in kwargs:
                    if self.motion_mode == "jtj" or self.motion_mode == "ctc":
                        self.vector_v_max = np.ones(np.shape(self.start_vector)) * self.v_max
                    elif self.motion_mode == "ptp":
                        self.vector_v_max = (
                            np.ones(np.shape(self.start_vector))
                            * self.v_max
                            * np.abs(self.end_vector - self.start_vector)
                            / np.linalg.norm(self.end_vector - self.start_vector)
                        )

                else:
                    if (
                        len(np.array(self.vector_v_max).reshape(-1))
                        != vector_length_limit
                    ):
                        raise ValueError(
                            "The size of vector_v_max doesn't match with vector_length_limit: ({vector_length_limit})"
                        )

            if "a_max" in kwargs:
                if "vector_a_max" not in kwargs:
                    if self.motion_mode == "jtj" or self.motion_mode == "ctc":
                        self.vector_a_max = np.ones(np.shape(self.start_vector)) * self.a_max
                    elif self.motion_mode == "ptp":
                        self.vector_a_max = (
                            np.ones(np.shape(self.start_vector))
                            * self.a_max
                            * np.abs(self.end_vector - self.start_vector)
                            / np.linalg.norm(self.end_vector - self.start_vector)
                        )

                else:
                    if (
                        len(np.array(self.vector_a_max).reshape(-1))
                        != vector_length_limit
                    ):
                        raise ValueError(
                            "The size of vector_a_max doesn't match with vector_length_limit: ({vector_length_limit})"
                        )
                        
        self.ctc_start_config = kwargs.get("ctc_start_config", 0)
        self.ctc_end_config = kwargs.get("ctc_end_config", 0)

        self.forward_kinematic_check = kwargs.get("forward_kinematic_check", False)
        self.plots = kwargs.get("plots", False)

        return True

    def profile_required_time_calculator(
        self,
        start_vector,
        end_vector,
        vector_v_max,
        vector_a_max,
        motion_mode,
        motion_profile,
        sync=True,
        degrees=False,
        ctc_start_config=0,
        ctc_end_config=0,
    ):

        def trapezoidal_vector_time_calcualtor(data):
            """
            Compute acceleration and total time for a trapezoidal motion profile
            given a distance, max velocity, and max acceleration.
            """
            d, v, a = np.split(data, 3)

            if d == 0:
                return np.reshape([0, 0], -1)

            d_acc = 0.5 * v**2 / a
            if d > 2 * d_acc:
                t_acc = v / a
                t_cruise = (d - 2 * d_acc) / v
                total_time = 2 * t_acc + t_cruise
            else:
                t_acc = np.sqrt(d / a)
                total_time = 2 * t_acc
            return np.reshape([t_acc, total_time], -1)

        def cycloidal_vector_time_calcualtor(data):
            """
            Compute acceleration and total time for a cycloidal motion profile
            given a distance, max velocity, and max acceleration.
            """
            d, v, a = np.split(data, 3)

            if d == 0:
                return np.reshape([0, 0], -1)

            required_time_base_on_accelerations = np.sqrt(2 * np.pi * d / a)
            required_time_base_on_velocities = 2 * d / v

            total_time = np.max(
                [required_time_base_on_accelerations, required_time_base_on_velocities],
                axis=0,
            )
            t_acc = total_time

            return np.reshape([t_acc, total_time], -1)

        delta = np.array(end_vector).reshape(-1) - np.array(start_vector).reshape(-1)

        if motion_mode == "jtj":
            displacement_vector = np.abs(np.reshape(delta, (-1, 1)))
            vector_v_max = np.reshape(vector_v_max, np.shape(displacement_vector))
            vector_a_max = np.reshape(vector_a_max, np.shape(displacement_vector))
            data = np.hstack([displacement_vector, vector_v_max, vector_a_max])

            if motion_profile == "trapezoidal":
                # Compute max time over all axes
                times_array = np.array(
                    list(map(trapezoidal_vector_time_calcualtor, data))
                )

            elif motion_profile == "cycloidal":
                # Compute max time over all axes
                times_array = np.array(
                    list(map(cycloidal_vector_time_calcualtor, data))
                )

        if motion_mode == "ptp":

            if self.num_freedom > 3:
                if degrees:
                    distance = np.concatenate([delta[:3], np.deg2rad(delta[3:])])
                else:
                    distance = np.array(delta).reshape(-1)
            else:
                distance = delta

            distance = np.resize(distance, 6)
            vector_v_max = np.resize(vector_v_max, 6)
            vector_a_max = np.resize(vector_a_max, 6)

            distance[len(delta) :] = 0
            vector_v_max[len(delta) :] = 1
            vector_a_max[len(delta) :] = 1

            displacement_vector = np.abs(np.reshape(distance, (6, 1)))
            vector_v_max = np.reshape(vector_v_max, np.shape(displacement_vector))
            vector_a_max = np.reshape(vector_a_max, np.shape(displacement_vector))
            data = np.hstack([displacement_vector, vector_v_max, vector_a_max])

            if motion_profile == "trapezoidal":
                # Compute max time over all axes
                times_array = np.array(
                    list(map(trapezoidal_vector_time_calcualtor, data))
                )

            elif motion_profile == "cycloidal":
                # Compute max time over all axes
                times_array = np.array(
                    list(map(cycloidal_vector_time_calcualtor, data))
                )
        
        elif motion_mode == "ctc":
            
            ee_pose = self.arm_kin.make_ee_pose(start_vector, degree=degrees)
            start_vector = self.arm_kin.closest_solution(ee_pose=ee_pose, config_num=ctc_start_config)
            self.start_vector = start_vector
            
            ee_pose = self.arm_kin.make_ee_pose(end_vector, degree=degrees)
            end_vector = self.arm_kin.closest_solution(ee_pose=ee_pose, config_num=ctc_end_config)
            self.end_vector = end_vector
                        
            delta = np.array(end_vector).reshape(-1) - np.array(start_vector).reshape(-1)

            displacement_vector = np.abs(np.reshape(delta, (-1, 1)))
            vector_v_max = np.reshape(vector_v_max, np.shape(displacement_vector))
            vector_a_max = np.reshape(vector_a_max, np.shape(displacement_vector))
            data = np.hstack([displacement_vector, vector_v_max, vector_a_max])

            if motion_profile == "trapezoidal":
                # Compute max time over all axes
                times_array = np.array(
                    list(map(trapezoidal_vector_time_calcualtor, data))
                )

            elif motion_profile == "cycloidal":
                # Compute max time over all axes
                times_array = np.array(
                    list(map(cycloidal_vector_time_calcualtor, data))
                )      
 
        else:
            raise NameError
        
        
        max_acc_times, max_total_times = times_array[:, 0], times_array[:, 1]
        if sync == False:
            times = {"accelerating_time": max_acc_times, "total_time": max_total_times}
        else:
            syn_max_total_time = np.full_like(max_total_times, 1) * np.max(
                max_total_times
            )
            sync_max_acc_time = (
                np.full_like(max_acc_times, 1)
                * max_acc_times[np.argmax(max_total_times)]
            )
            times = {
                "accelerating_time": sync_max_acc_time,
                "total_time": syn_max_total_time,
            }

        # print(times)
        return times

    def generate_profile_over_time(self, motion_profile, times, num, window_size):

        def trapezoidal_profile(data):
            """
            Generate normalized trapezoidal profile s(t) ∈ [0, 1]
            over a given total time and acceleration time.
            """
            total_time, accel_time, max_total_time, num = data[0:4]
            num = int(num) if num != 0 else 100

            t = np.linspace(0, max_total_time, num)
            v_max = (
                1 / (total_time - accel_time) if (total_time - accel_time) != 0 else 0
            )

            s = np.full_like(t, 1)

            for i, ti in enumerate(t):
                if not (total_time == 0 and accel_time == 0):
                    if ti <= total_time:
                        if ti < accel_time:
                            s[i] = 0.5 * v_max / accel_time * ti**2
                        elif ti < total_time - accel_time:
                            s[i] = 0.5 * v_max * accel_time + v_max * (ti - accel_time)
                        else:
                            td = total_time - ti
                            s[i] = 1 - 0.5 * v_max / accel_time * td**2
                else:
                    s[i] = 1

            return s

        def cycloidal_profile(data):
            """
            Generate normalized cycloidal profile s(t) ∈ [0, 1]
            over a given total time and acceleration time.
            """
            total_time, accel_time, max_total_time, num = data[0:4]
            num = int(num) if num != 0 else 100

            t = np.linspace(0, max_total_time, num)
            s = np.full_like(t, 1)

            for i, ti in enumerate(t):
                if not (total_time == 0 and accel_time == 0):
                    if ti <= total_time:
                        s[i] = (ti / total_time) - (1 / (2 * np.pi)) * np.sin(
                            2 * np.pi * ti / total_time
                        )
                else:
                    s[i] = 1

            return s

        minimum_total_time = np.min([x for x in times["total_time"] if x > 0])
        maximum_total_time = np.max(times["total_time"])
        num = max(num, int(maximum_total_time / (minimum_total_time / window_size)) + 1)
        t = np.linspace(0, maximum_total_time, num)

        total_time_vector = np.reshape(times["total_time"], (-1, 1))
        accelerating_time_vector = np.reshape(times["accelerating_time"], (-1, 1))
        maximum_total_time_vector = np.full_like(total_time_vector, maximum_total_time)
        num_vector = np.full_like(total_time_vector, num)

        vector_size = len(total_time_vector)

        data = np.hstack(
            [
                total_time_vector,
                accelerating_time_vector,
                maximum_total_time_vector,
                num_vector,
            ]
        )

        if motion_profile == "trapezoidal":
            profiles_list = list(map(trapezoidal_profile, data))
            profiles = np.split(np.array(profiles_list), vector_size, axis=0)

        elif motion_profile == "cycloidal":
            profiles_list = list(map(cycloidal_profile, data))
            profiles = np.split(np.array(profiles_list), vector_size, axis=0)

        profiles_in_order = np.hstack(
            [np.array(profiles[i]).reshape(-1, 1) for i in range(vector_size)]
        )
        return t, profiles_in_order

    def joints_profile_generator(
        self,
        start_vector,
        end_vector,
        s_profiles,
        t,
        motion_mode,
        motion_profile,
        window_length,
        poly_order,
        arm_config=None,
    ):

        def interpolate_vector(start_vector, end_vector, s):
            """
            Linearly interpolate between start and end pose for each component
            using scalar profile s(t).
            """
            start_vector = np.array(start_vector)
            end_vector = np.array(end_vector)
            s = np.array(s)
            interpolated = (1 - s) * start_vector + s * end_vector
            return interpolated

        if motion_mode == "jtj" or motion_mode == "ctc":
            profiles = interpolate_vector(start_vector, end_vector, s_profiles)
            joints_profile_vector = profiles

        elif motion_mode == "ptp":
            augmented_start_vector = np.resize(start_vector, 6)
            augmented_end_vector = np.resize(end_vector, 6)
            augmented_start_vector[len(start_vector) :] = 0
            augmented_end_vector[len(end_vector) :] = 0

            start_vector = augmented_start_vector
            end_vector = augmented_end_vector

            profiles = interpolate_vector(start_vector, end_vector, s_profiles)


            class calc_inverse_kinematics():

                def __init__(self, arm_kin: Kinematics):
                    self.prev_joint = None
                    self.singularities  = 0
                    self.arm_kin = arm_kin
                def __call__(self, ee_pose, config_num, singularities_limit = 10,flush = False):
                    try:
                        joint_angles = self.arm_kin.closest_solution(ee_pose, current_joint_angles = self.prev_joint, config_num = config_num)
                        self.prev_joint = joint_angles if not flush else None

                        if self.singularities > singularities_limit:
                            raise ValueError(
                                            f"Number of singular points ({self.singularities}) exceed the limit ({singularities_limit}) for this point{ee_pose}"
                                            )

                        return joint_angles
                    except RuntimeError:
                        self.singularities += 1
                        return self.prev_joint
                    except Exception as e:
                        raise ValueError(
                        f"Faced Error! {e}"
                        )
                    
            calc_inverse =  calc_inverse_kinematics(self.arm_kin)

            ee_pose = np.array(list(map(self.arm_kin.make_ee_pose, profiles)))
            partial_func = partial(calc_inverse, config_num=arm_config, singularities_limit = self.singularities_limit)
            joints_profile_vector = np.array(list(map(partial_func, ee_pose)))

        dt = t[1] - t[0]
        jths = joints_profile_vector
        djths = savgol_filter(
            jths, window_length, polyorder=poly_order, deriv=1, delta=dt, axis=0
        )
        ddjths = savgol_filter(
            jths, window_length, polyorder=poly_order, deriv=2, delta=dt, axis=0
        )

        return jths, djths, ddjths

    def augment_joints_profile(self, arm_num_joints, num_joints, jths, djths, ddjths):
        num_rows = np.shape(jths)[0]
        if arm_num_joints > num_joints:
            augmentation_vectors = np.zeros(
                (num_rows, self.arm_num_joints - self.num_joints)
            )
            augmented_jths = np.hstack([jths, augmentation_vectors])
            augmented_djths = np.hstack([djths, augmentation_vectors])
            augmented_ddjths = np.hstack([ddjths, augmentation_vectors])
        else:
            if arm_num_joints < num_joints:
                print(
                    f"Be careful some motion might not perform because arm_num_joints ({arm_num_joints}) < num_joints({num_joints})!"
                )
            augmented_jths = jths[:, :arm_num_joints]
            augmented_djths = djths[:, :arm_num_joints]
            augmented_ddjths = ddjths[:, :arm_num_joints]

        return augmented_jths, augmented_djths, augmented_ddjths

    def calc_end_effector_profiles(self, jths, window_length, poly_order, dt):
        p = np.array(list(map(self.arm_kin.forward, jths)))[:, :3, 3]
        dp = savgol_filter(
            p, window_length, polyorder=poly_order, deriv=1, delta=dt, axis=0
        )
        ddp = savgol_filter(
            p, window_length, polyorder=poly_order, deriv=2, delta=dt, axis=0
        )
        return p, dp, ddp

    def calc_torqes(self, data):  # type: ignore

        jth, djth, ddjth = np.split(data, 3)
        p, dp, ddp, w, alpha, taus = self.arm_dyn.inverse(jth, djth, ddjth)  # type: ignore
        return taus

    def compute_energy(
        self,
        start_vector,
        end_vector,
        motion_mode,
        check_keyword=True,
        plots=False,
        **kwargs,
    ):

        keywords = {
            **kwargs,
            "start_vector": start_vector,
            "end_vector": end_vector,
            "motion_mode": motion_mode,
            "plots": plots,
        }

        self.assign_and_check_keywords(check_keyword, **keywords)

        # Process initiated!
        process_start_time = time.time()

        times = self.profile_required_time_calculator(
            self.start_vector,
            self.end_vector,
            self.vector_v_max,
            self.vector_a_max,
            self.motion_mode,
            self.motion_profile,
            self.sync,
            ctc_start_config = self.ctc_start_config,
            ctc_end_config = self.ctc_end_config,
        )
        
        if times['total_time'].all() == 0:
            return 0, 0
        
        t, profiles = self.generate_profile_over_time(
            self.motion_profile, times, self.num, self.window_length
        )
        jths, djths, ddjths = self.joints_profile_generator(self.start_vector, self.end_vector, profiles, t, self.motion_mode, self.motion_profile, self.window_length, self.poly_order, self.arm_config)  # type: ignore
        jths, djths, ddjths = self.augment_joints_profile(
            self.arm_num_joints, self.num_joints, jths, djths, ddjths
        )

        taus = np.array(list(map(self.calc_torqes, np.hstack([jths, djths, ddjths]))))

        power = np.sum(np.maximum(djths * taus, 0), axis=1)
        dt = t[1] - t[0]
        energy = np.trapz(power, dx=dt)

        # Process ended!
        process_end_time = time.time()

        if self.forward_kinematic_check:
            p, dp, ddp = self.calc_end_effector_profiles(
                jths, self.window_length, self.poly_order, dt
            )

        if self.plots:

            plt.figure(1)
            plt.title("joints position vs time")
            plt.plot(t, jths)

            plt.figure(2)
            plt.title("joints velocities vs time")
            plt.plot(t, djths)

            plt.figure(3)
            plt.title("joints acceleration vs time")
            plt.plot(t, ddjths)

            plt.figure(4)
            plt.title("joints toques vs time")
            plt.plot(t, taus)

            plt.figure(5)
            plt.title("Power vs time")
            plt.plot(t, power)

            if self.forward_kinematic_check:
                fig = plt.figure(6)
                ax = fig.add_subplot(111, projection="3d")
                ax.set_title("end effector position vs time")
                ax.plot(p[:, 0], p[:, 1], p[:, 2])
                ax.scatter(p[0, 0], p[0, 1], p[0, 2], c="k")
                ax.scatter(p[-1, 0], p[-1, 1], p[-1, 2], c="k")

                plt.figure(7)
                plt.title("end effector velocities vs time")
                plt.plot(t, dp)

                plt.figure(8)
                plt.title("end effector acceleration vs time")
                plt.plot(t, ddp)

                plt.figure(9)
                plt.title("end effector position vs time")
                plt.plot(t, p)

                plt.figure(10)
                plt.title("end effector velocities vs time")
                plt.plot(t, np.linalg.norm(dp, axis=1))

                plt.figure(11)
                plt.title("end effector acceleration vs time")
                plt.plot(t, np.linalg.norm(ddp, axis=1))

            plt.show()

        self.process_time = process_end_time - process_start_time

        return self.process_time, energy


if __name__ == "__main__":
    ur_arm = Arm("ur5e")

    for i in range(1):
        kwargs = {
            "v_max": 0.2,
            "a_max": 0.6,
            "profile": "trapezoidal",  # trapezoidal, cycloidal
            "num": 100,
            "sync": True,
            "num_freedom": 6,
            "plots": True,
            "forward_kinematic_check": True,
            "arm_config": 3,
            "singularities_limit": 5,
            "ctc_start_config": 0,
            "ctc_end_config": 7,
        }
        
        # motion_mode: 'jtj', 'ptp', 'ctc' 


        start_pos = [-0.5, 0.1, 0.0, np.pi/2, 0.1, np.pi - 0.1]
        # end_pos = [-0.5, 0.1, 0.0, np.pi/2, 0.1, np.pi - 0.1] # 
        end_pos = [-0.3, -0.1, 0.3, np.pi/2, 0.1, np.pi - 0.1]
        
        results = ur_arm.compute_energy(
            start_vector=start_pos, end_vector=end_pos, motion_mode="ptp", **kwargs
        )

        print(results)
