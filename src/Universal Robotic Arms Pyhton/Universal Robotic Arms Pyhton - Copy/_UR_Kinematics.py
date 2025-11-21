import numpy as np
from numpy import sin, cos, arctan2, arccos, arcsin, deg2rad, pi
from scipy.spatial.transform import Rotation as R
import yaml
import time
import os
from collections import defaultdict, deque

class Kinematics():

    def __init__(self, arm = 'ur5'):

        file_path = 'arms_config.yaml'
            
        try:
            with open(file_path, 'r') as f:
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

    def forward(self, thetas, degree = False):
        if degree:
            thetas = deg2rad(thetas)

        r11 = cos(thetas[5])*(sin(thetas[0])*sin(thetas[4]) + cos(thetas[1] + thetas[2] + thetas[3])*cos(thetas[0])*cos(thetas[4])) - 1.0*sin(thetas[1] + thetas[2] + thetas[3])*cos(thetas[0])*sin(thetas[5])
        r12 = - 1.0*sin(thetas[5])*(sin(thetas[0])*sin(thetas[4]) + cos(thetas[1] + thetas[2] + thetas[3])*cos(thetas[0])*cos(thetas[4])) - 1.0*sin(thetas[1] + thetas[2] + thetas[3])*cos(thetas[0])*cos(thetas[5])
        r13 = cos(thetas[4])*sin(thetas[0]) - 1.0*cos(thetas[1] + thetas[2] + thetas[3])*cos(thetas[0])*sin(thetas[4])
        r21 = - 1.0*cos(thetas[5])*(cos(thetas[0])*sin(thetas[4]) - 1.0*cos(thetas[1] + thetas[2] + thetas[3])*cos(thetas[4])*sin(thetas[0])) - 1.0*sin(thetas[1] + thetas[2] + thetas[3])*sin(thetas[0])*sin(thetas[5])
        r22 = sin(thetas[5])*(cos(thetas[0])*sin(thetas[4]) - 1.0*cos(thetas[1] + thetas[2] + thetas[3])*cos(thetas[4])*sin(thetas[0])) - 1.0*sin(thetas[1] + thetas[2] + thetas[3])*cos(thetas[5])*sin(thetas[0])
        r23 = - 1.0*cos(thetas[0])*cos(thetas[4]) - 1.0*cos(thetas[1] + thetas[2] + thetas[3])*sin(thetas[0])*sin(thetas[4])
        r31 = cos(thetas[1] + thetas[2] + thetas[3])*sin(thetas[5]) + sin(thetas[1] + thetas[2] + thetas[3])*cos(thetas[4])*cos(thetas[5])
        r32 = cos(thetas[1] + thetas[2] + thetas[3])*cos(thetas[5]) - 1.0*sin(thetas[1] + thetas[2] + thetas[3])*cos(thetas[4])*sin(thetas[5])
        r33 = -1.0*sin(thetas[1] + thetas[2] + thetas[3])*sin(thetas[4])

        x = self.d4*sin(thetas[0]) + self.a2*cos(thetas[0])*cos(thetas[1]) + self.d6*cos(thetas[4])*sin(thetas[0]) + - self.a3*cos(thetas[0])*sin(thetas[1])*sin(thetas[2]) - self.d6*cos(thetas[1] + thetas[2] + thetas[3])*cos(thetas[0])*sin(thetas[4]) + self.d5*cos(thetas[1] + thetas[2])*cos(thetas[0])*sin(thetas[3]) + self.d5*sin(thetas[1] + thetas[2])*cos(thetas[0])*cos(thetas[3]) + self.a3*cos(thetas[0])*cos(thetas[1])*cos(thetas[2])
        y = - self.a3*sin(thetas[0])*sin(thetas[1])*sin(thetas[2]) - self.d6*cos(thetas[0])*cos(thetas[4]) + self.a2*cos(thetas[1])*sin(thetas[0]) - self.d4*cos(thetas[0]) - self.d6*cos(thetas[1] + thetas[2] + thetas[3])*sin(thetas[0])*sin(thetas[4]) + self.d5*cos(thetas[1] + thetas[2])*sin(thetas[0])*sin(thetas[3]) + self.d5*sin(thetas[1] + thetas[2])*cos(thetas[3])*sin(thetas[0]) + self.a3*cos(thetas[1])*cos(thetas[2])*sin(thetas[0])
        z = self.d5*sin(thetas[1] + thetas[2])*sin(thetas[3]) + self.a2*sin(thetas[1]) - - self.a3*sin(thetas[1] + thetas[2]) - 1.0*sin(thetas[4])*(self.d6*cos(thetas[1] + thetas[2])*sin(thetas[3]) + self.d6*sin(thetas[1] + thetas[2])*cos(thetas[3])) - self.d5*cos(thetas[1] + thetas[2])*cos(thetas[3]) + self.d1

        ee_pose = np.eye(4)
        ee_pose[0:3, 0:3] = np.array([[r11, r12, r13],
                                      [r21, r22, r23],
                                      [r31, r32, r33]])
        ee_pose[0:3, 3] = np.array([x, y, z])
        return ee_pose
     
    def inverse(self, ee_pose):

        euler_angles =  R.from_matrix(ee_pose[0:3, 0:3]).as_euler('zyx', degrees=False)

        T06 = np.vstack((ee_pose, [0, 0, 0, 1]))
        P06 = T06[:3, 3]
        P05 = np.dot(T06 ,np.array([0, 0, -self.d6, 1]).T)

        #--------------------- theta 1 ---------------------
        th1 = np.full(2, None, float)
        try:
            th1[:]  = arctan2(P05[1],P05[0]) + pi/2

            if P05[0] != 0 and P05[1] != 0:
                th1[0] += - np.real(np.arccos(self.d4/ np.linalg.norm(P05[:2])))
                th1[1] += + np.real(np.arccos(self.d4/ np.linalg.norm(P05[:2]))) 

            for th in np.nditer(th1, op_flags=['readwrite']):
                if np.abs(th) > pi:
                    th[...] += - 2 * pi

            #print(th1)
        except:
            pass

        #--------------------- theta 5 ---------------------
        th5 = np.full(4, None, float)
        try:
            i = 0
            for th_1 in th1:
                arccosValue = ((np.dot(P06[0:2], np.array([sin(th_1), -cos(th_1)]).T) - self.d4)/self.d6)
                if np.abs(arccosValue) < 1:
                    th5[i] = +1 * arccos(arccosValue)
                    th5[i+1] = -1 * arccos(arccosValue)
                
                i +=2

            #print(th5)
        except:
            pass

        #--------------------- theta 6 ---------------------
        def calculateTheta6(X60, Y60, th_1, th_5):
            th_6 = None
            try:
                if sin(th_5) != 0:
                    leftNumerator = -X60[1]*sin(th_1) + Y60[1]*cos(th_1)
                    rightNumerator = X60[0]*sin(th_1) - Y60[0]*cos(th_1)
                    denominator = sin(th_5)
                    th_6 = arctan2(leftNumerator/denominator, rightNumerator/denominator)
            except:
                pass
            return th_6
        
        th6 = np.full(4, None, float)
        try:
            T60 = np.linalg.inv(T06)
            X60 = T60[:3,0]
            Y60 = T60[:3,1]

            t = [0, 2]
            i = 0
            for t1 in range(2):
                th6[i] = calculateTheta6(X60, Y60, th1[t1], th5[t[t1]])
                th6[i+1] = calculateTheta6(X60, Y60, th1[t1], th5[t[t1]+1])
                i += 2

            #print(th6)
        except:
            pass

        #--------------------- theta 3 ---------------------
        def DH2tform(alpha, a, d, theta):
            Transform = np.eye(4)

            Transform[0, 0] = cos(theta)
            Transform[0, 1] = -sin(theta)
            Transform[0, 2] = 0
            Transform[0, 3] = a

            Transform[1, 0] = sin(theta)*cos(alpha)
            Transform[1, 1] = cos(theta) *cos(alpha)
            Transform[1, 2] = -sin(alpha)
            Transform[1, 3] = -sin(alpha)*d

            Transform[2, 0] = sin(theta)*sin(alpha)
            Transform[2, 1] = cos(theta)*sin(alpha)
            Transform[2, 2] = cos(alpha)
            Transform[2, 3] = cos(alpha)*d

            return Transform

        def calculateP14(T06, th_1, th_5, th_6):

            T01 = DH2tform(0, 0, self.d1, th_1)
            T10 = np.linalg.inv(T01)

            T45 = DH2tform(self.alpha4, self.a4, self.d5, th_5)
            T54 = np.linalg.inv(T45)

            T56 = DH2tform(self.alpha5, self.a5, self.d6, th_6)
            T65 = np.linalg.inv(T56)

            T14 = T10@T06@T65@T54
            P14 = T14[:3, 3]

            return P14, T14
            
        def calculateTheta3(T06, theta1, theta5, theta6):
            P14, T14 = calculateP14(T06, theta1, theta5, theta6)
            P14_xz_length = np.linalg.norm([P14[0], P14[2]])

            conditions = [abs(self.a2-self.a3),
                          abs(self.a2+self.a3)]
            
            if P14_xz_length > conditions[0] and P14_xz_length < conditions[1]:
                theta3 = arccos((P14_xz_length**2 - self.a2**2 -self.a3**2)/(2*self.a2*self.a3))
            else:
                pass

            return theta3

        th3 = np.full(8, None, float)
        signs = [+1, -1]
        for i in range(len(th3)):
            try:
                th_1 = th1[int(i/4)]
                th_5 = th5[int(i/2)]
                th_6 = th6[int(i/2)]
                sign = signs[int(i%2)]
                theta3 = sign * calculateTheta3(T06, th_1, th_5, th_6)
                th3[i] = theta3
            except:
                pass
        
        #print(th3)
        #--------------------- theta 2 ---------------------  
        def calculateTheta2(T06, theta1, theta3, theta5, theta6):
            P14, T14 = calculateP14(T06, theta1, theta5, theta6)
            P14_xz_length = np.linalg.norm([P14[0], P14[2]])

            theta2 = arctan2(-P14[2], -P14[0]) - arcsin(-self.a3 * sin(theta3)/P14_xz_length)
            if abs(theta2) > pi:
                if theta2 > 0:
                    theta2 += -2*pi
                else:
                    theta2 += +2*pi
            return theta2

        th2 = np.full(8, None, float)
        for i in range(len(th2)):
            try:
                th_1 = th1[int(i/4)]
                th_5 = th5[int(i/2)]
                th_6 = th6[int(i/2)]
                th_3 = th3[i]
                theta2 = calculateTheta2(T06, th_1, th_3, th_5, th_6)
                #print(np.round([th_1, theta2, th_3, th_5, th_6], 4))
                th2[i] = theta2
            except:
                pass
        #print(th2)

        #--------------------- theta 4 ---------------------
        def calculateTheta4(theta1, theta2, theta3, theta5, theta6):

            P14, T14 = calculateP14(T06, theta1, theta5, theta6)
            
            T12 = DH2tform(self.alpha1, self.a1, self.d2, theta2)
            T21 = np.linalg.inv(T12)

            T23 = DH2tform(self.alpha2, self.a2, self.d3, theta3)
            T32 = np.linalg.inv(T23)

            T34 = T32@T21@T14
            X34 = T34[:3, 0]

            theta4 = arctan2(X34[1], X34[0])
            return theta4
   
        th4 = np.full(8, None, float)
        for i in range(8):
            try:
                th_1 = th1[int(i/4)]
                th_2 = th2[i]
                th_3 = th3[i]
                th_5 = th5[int(i/2)]
                th_6 = th6[int(i/2)]
                theta4 = calculateTheta4(th_1, th_2, th_3, th_5, th_6)
                #print(np.round([th_1, th_2, th_3, theta4, th_5, th_6], 4))
                th4[i] = theta4
            except:
                pass
        #print(th4)


        #--------------------- sort solutoins ---------------------
        def sorted_solution(th1, th2, th3, th4, th5, th6):
            configs = defaultdict(list)

            for i in range(8):
                th_1 = th1[int(i/4)]
                th_2 = th2[i]
                th_3 = th3[i]
                th_4 = th4[i]
                th_5 = th5[int(i/2)]
                th_6 = th6[int(i/2)]

                configs[str(i)] = [th_1, th_2, th_3, th_4, th_5, th_6]

            sorted_configs = [configs['3'],
                              configs['7'],
                              configs['1'],
                              configs['5'],
                              configs['0'],
                              configs['4'],
                              configs['6'],
                              configs['2']]

            #sorted_configs = configs
            return sorted_configs
        

        def feasibility_check(configs):
            feasibility = np.full(8, False)
            feasible_configs = []
            counter = 0
            for config in configs:
                if np.isnan(config).any():
                    #print('nan value:',config)
                    pass
                else:
                    if abs(np.mean(self.forward(config)[:3, :]- ee_pose)) < 1e-4:
                        feasible_configs.append(config)
                        feasibility[counter] = True
                counter +=1
                #print(feasibility, np.array(feasible_configs))

            return feasibility, configs, np.array(feasible_configs)

        configs = sorted_solution(th1, th2, th3, th4, th5, th6)
        f, c, fc = feasibility_check(configs)
        return f, c, fc
    
        
    def make_ee_pose(self, position_vector, degree=False):
        position_vector = np.array(position_vector).reshape(-1)
        vector_size = len(position_vector)
        if vector_size <= 6:
            position_vector = np.resize(position_vector, 6)
            position_vector[vector_size:] = 0
        else:
           position_vector = position_vector[:6] 
        translation_vector, orientation_vector = np.split(position_vector, 2)
        orientation_matrix = R.from_euler("zyx", orientation_vector, degrees=degree).as_matrix()
        translation = np.reshape(translation_vector[0:3], (3,1))
        ee_pose = np.hstack((orientation_matrix, translation))
        return ee_pose

    def closest_solution(self, ee_pose, current_joint_angles=None, config_num = None):
            _, _, joint_configs = self.inverse(ee_pose)
            if not joint_configs.any():
                raise RuntimeError(f"No IK solution found for this end effector position {ee_pose}")
            
            current_joint_angles = self.current_joint_angles
            
            if current_joint_angles is not None:
                diffs = joint_configs - current_joint_angles

                diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
                distances = np.linalg.norm(diffs, axis=1)
                config = np.argmin(distances)
            else:
                config = 0
                pass
            
            if config_num is not None:
                config = config_num

            selected = joint_configs[config]
            if current_joint_angles is not None:
                selected = (selected - current_joint_angles + np.pi) % (2 * np.pi) - np.pi + current_joint_angles

            self.current_joint_angles = selected
            return np.round(selected, 5)



if __name__ == '__main__':
    pass
