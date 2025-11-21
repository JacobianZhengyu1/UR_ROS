import numpy as np
from numpy import sin, cos, arctan2, arccos, arcsin, deg2rad, pi
from scipy.spatial.transform import Rotation as R
import yaml
import time
import os
from collections import defaultdict, deque
import matplotlib.pyplot as plt


class Dynamics():
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

        self.arm_config = self.configs[arm]
        self.dh_parameters = self.arm_config['dh']
        self.inertia_parameters = self.arm_config['inertia']

        # --------------- dh parameters ---------------

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


        # --------------- inertia parameters ---------------

        self.m1 = self.inertia_parameters['m1']
        self.m2 = self.inertia_parameters['m2']
        self.m3 = 0.0
        self.m4 = self.inertia_parameters['m3']
        self.m5 = 0.0
        self.m6 = self.inertia_parameters['m4']
        self.m7 = self.inertia_parameters['m5']
        self.m8 = self.inertia_parameters['m6']

        self.Ic11 = np.diag(self.inertia_parameters['I1'])
        self.Ic22 = np.diag(self.inertia_parameters['I2'])
        self.Ic33 = np.zeros((3,3))
        self.Ic44 = np.diag(self.inertia_parameters['I3'])
        self.Ic55 = np.zeros((3,3))
        self.Ic66 = np.diag(self.inertia_parameters['I4'])
        self.Ic77 = np.diag(self.inertia_parameters['I5'])
        self.Ic88 = np.diag(self.inertia_parameters['I6'])

        self.Pc11 = np.array(self.inertia_parameters['Pc1']).T
        self.Pc22 = np.array(self.inertia_parameters['Pc2']).T
        self.Pc33 = np.array([0.0, 0.0, 0.0]).T
        self.Pc44 = np.array(self.inertia_parameters['Pc3']).T
        self.Pc55 = np.array([0.0, 0.0, 0.0]).T
        self.Pc66 = np.array(self.inertia_parameters['Pc4']).T
        self.Pc77 = np.array(self.inertia_parameters['Pc5']).T
        self.Pc88 = np.array(self.inertia_parameters['Pc6']).T

        self.g = -9.81

    def inverse(self, thetas, dthetas = np.zeros(6), ddthetas = np.zeros(6)):

        def T(a, alpha, theta, d):
            """
            Compute the homogeneous transformation matrix using standard DH parameters.
            Parameters:
                a:     Link length
                alpha: Link twist
                theta: Joint angle
                d:     Link offset
            Returns:
                4x4 numpy array representing the transformation matrix
            """
            T1 = np.array(
                [[1, 0, 0, 0],
                [0, cos(alpha), -sin(alpha), 0],
                [0, sin(alpha), cos(alpha), 0],
                [0, 0, 0, 1]]
                )

            T2 = np.array(
                [[1, 0, 0, a],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
                )

            T3 = np.array(
                [[cos(theta), -sin(theta), 0, 0],
                [sin(theta), cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
                )

            T4 = np.array(
                [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, d],
                [0, 0, 0, 1]]
                )

            return (T1 @ T2 @ T3 @T4)


        T01 = T(0, 0, thetas[0], self.d1)
        T12 = T(0, pi/2, thetas[1], self.d4)
        T23 = T(self.a2, 0, 0, 0)
        T34 = T(0, 0, thetas[2], -self.d4)
        T45 = T(self.a3, pi, 0, 0)
        T56 = T(0, pi, thetas[3], self.d4)
        T67 = T(0, pi/2, thetas[4], self.d5)
        T78 = T(0,  -pi/2, thetas[5], self.d6)

        T02 = T01 @ T12
        T03 = T02 @ T23
        T04 = T03 @ T34
        T05 = T04 @ T45
        T06 = T05 @ T56
        T07 = T06 @ T67
        T08 = T07 @ T78

       
        R01 = T01[:3 , :3]
        R10 = R01.T
        R12 = T12[:3 , :3]
        R21 = R12.T
        R23 = T23[:3 , :3]
        R32 = R23.T
        R34 = T34[:3 , :3]
        R43 = R34.T
        R45 = T45[:3 , :3]
        R54 = R45.T
        R56 = T56[:3 , :3]
        R65 = R56.T
        R67 = T67[:3 , :3]
        R76 = R67.T
        R78 = T78[:3 , :3]
        R87 = R78.T

        P01 = T01[:3 , 3]
        P12 = T12[:3 , 3]
        P23 = T23[:3 , 3]
        P34 = T34[:3 , 3]
        P45 = T45[:3 , 3]
        P56 = T56[:3 , 3]
        P67 = T67[:3 , 3]
        P78 = T78[:3 , 3]

        P08 = T08[:3 , 3]

        R02 = R01 @ R12
        R03 = R02 @ R23
        R04 = R03 @ R34
        R05 = R04 @ R45
        R06 = R05 @ R56
        R07 = R06 @ R67
        R08 = T08[:3 , :3]

        P01 = T01[:3 , 3]
        P12 = T12[:3 , 3]
        P23 = T23[:3 , 3]
        P34 = T34[:3 , 3]
        P45 = T45[:3 , 3]
        P56 = T56[:3 , 3]
        P67 = T67[:3 , 3]
        P78 = T78[:3 , 3]

        v00 = np.array([0, 0, 0]).T
        w00 = np.array([0, 0, 0]).T
        a00 = np.array([0, 0, -self.g]).T
        alpha00 = np.array([0, 0, 0]).T

        w11 = R10 @ w00 + np.array([0, 0, dthetas[0]]).T
        v11 = R10 @ (v00 + np.cross(w00,P01))
        alpha11 = R10 @ alpha00 + np.cross(R10 @ w00, np.array([0, 0, dthetas[0]]).T) + np.array([0, 0, ddthetas[0]]).T 
        a11 = R10 @ (a00 + np.cross(alpha00, P01) + np.cross(w00, np.cross(w00, P01)))
        ac11 = a11 + np.cross(alpha11, self.Pc11) + np.cross(w11, np.cross(w11, self.Pc11))
        F11 = self.m1 * ac11
        N11 = self.Ic11 @ alpha11 + np.cross(w11, self.Ic11 @ w11)

        #print(F11, N11, sep='\n')

        w22 = R21 @ w11 + np.array([0, 0, dthetas[1]]).T
        v22 = R21 @ (v11 + np.cross(w11, P12))
        alpha22 = R21 @ alpha11 + np.cross(R21 @ w11, np.array([0, 0, dthetas[1]]).T) + np.array([0, 0, ddthetas[1]]).T 
        a22 = R21 @ (a11 + np.cross(alpha11, P12) + np.cross(w11, np.cross(w11, P12)))
        ac22 = a22 + np.cross(alpha22, self.Pc22) + np.cross(w22, np.cross(w22, self.Pc22))
        F22 = self.m2 * ac22
        N22 = self.Ic22 @ alpha22 + np.cross(w22, self.Ic22 @ w22)

        #print(F22, N22, sep='\n')
        
        w33 = R32 @ w22 + np.array([0, 0, 0]).T
        v33 = R32 @ (v22 + np.cross(w22,P23))
        alpha33 = R32 @ alpha22 + np.cross(R32 @ w22, np.array([0, 0, 0]).T) + np.array([0, 0, 0]).T
        a33 = R32 @ (a22 + np.cross(alpha22, P23) + np.cross(w22, np.cross(w22, P23)))
        F33 = np.array([0, 0, 0]).T
        N33 = np.array([0, 0, 0]).T

        #print(F33, N33, sep='\n')
        
        w44 = R43 @ w33 + np.array([0, 0, dthetas[2]]).T
        v44 = R43 @ (v33 + np.cross(w33, P34))
        alpha44 = R43 @ alpha33 + np.cross(R43 @ w33, np.array([0, 0, dthetas[2]]).T) + np.array([0, 0, ddthetas[2]]).T 
        a44 = R43 @ (a33 + np.cross(alpha33, P34) + np.cross(w33, np.cross(w33, P34)))
        ac44 = a44 + np.cross(alpha44, self.Pc44) + np.cross(w44, np.cross(w44, self.Pc44))
        F44 = self.m4 * ac44
        N44 = self.Ic44 @ alpha44 + np.cross(w44, self.Ic44 @ w44)

        #print(F44, N44, sep='\n')
        
        w55 = R54 @ w44 + np.array([0, 0, 0]).T
        v55 = R54 @ (v44 + np.cross(w44, P45))
        alpha55 = R54 @ alpha44 + np.cross(R54 @ w44, np.array([0, 0, 0]).T) + np.array([0, 0, 0]).T
        a55 = R54 @ (a44 + np.cross(alpha44, P45) + np.cross(w44, np.cross(w44, P45)))
        F55 = np.array([0, 0, 0]).T
        N55 = np.array([0, 0, 0]).T
        
        #print(F55, N55, sep='\n')
 
        w66 = R65 @ w55 + np.array([0, 0, dthetas[3]]).T
        v66 = R65 @ (v55 + np.cross(w55, P56))
        alpha66 = R65 @ alpha55 + np.cross(R65 @ w55, np.array([0, 0, dthetas[3]]).T) + np.array([0, 0, ddthetas[3]]).T 
        a66 = R65 @ (a55 + np.cross(alpha55, P56) + np.cross(w55, np.cross(w55, P56)))
        ac66 = a66 + np.cross(alpha66, self.Pc66) + np.cross(w66, np.cross(w66, self.Pc66))
        F66 = self.m6 * ac66
        N66 = self.Ic66 @ alpha66 + np.cross(w66, self.Ic66 @ w66)

        #print(F66, N66, sep='\n')

        w77 = R76 @ w66 + np.array([0, 0, dthetas[4]]).T
        v77 = R76 @ (v66 + np.cross(w66, P67))
        alpha77 = R76 @ alpha66 + np.cross(R76 @ w66, np.array([0, 0, dthetas[4]]).T) + np.array([0, 0, ddthetas[4]]).T 
        a77 = R76 @ (a66 + np.cross(alpha66, P67) + np.cross(w66, np.cross(w66, P67)))
        ac77 = a77 + np.cross(alpha77, self.Pc77) + np.cross(w77, np.cross(w77, self.Pc77))
        F77 = self.m7 * ac77
        N77 = self.Ic77 @ alpha77 + np.cross(w77, self.Ic77 @ w77)

        #print(F77, N77, sep='\n')

        w88 = R87 @ w77 + np.array([0, 0, dthetas[5]]).T
        v88 = R87 @ (v77 + np.cross(w77, P78))
        alpha88 = R87 @ alpha77 + np.cross(R87 @ w77, np.array([0, 0, dthetas[5]]).T) + np.array([0, 0, ddthetas[5]]).T 
        a88 = R87 @ (a77 + np.cross(alpha77, P78) + np.cross(w77, np.cross(w77, P78)))
        ac88 = a88 + np.cross(alpha88, self.Pc88) + np.cross(w88, np.cross(w88, self.Pc88))
        F88 = self.m8 * ac88
        N88 = self.Ic88 @ alpha88 + np.cross(w88, self.Ic88 @ w88)

        #print(F88, N88, sep='\n')

        v08 =  R08 @ v88
        a08 =  R08 @ a88

        w08 = R08 @ w88
        alpha08 = R08 @ alpha88




        f88 = np.array([0, 0, 0]).T + F88.T
        n88 = np.array([0, 0, 0]).T + N88.T

        f77 = R78 @ f88 + F77
        n77 = R78 @ n88 + np.cross(self.Pc77, F77) + np.cross(P78, R78 @ f88) + N77

        f66 = R67 @ f77 + F66
        n66 = R67 @ n77 + np.cross(self.Pc66, F66) + np.cross(P67, R67 @ f77) + N66

        f55 = R56 @ f66 + F55
        n55 = R56 @ n66 + np.cross(P56, R56 @  f66) + N55

        f44 = R45 @ f55 + F44
        n44 = R45 @ n55 + np.cross(self.Pc44, F44) + np.cross(P45, R45 @ f55) + N44

        f33 = R34 @ f44 + F33
        n33 = R34 @ n44 + np.cross(P34, R34 @ f44) + N33

        f22 = R23 @ f33 + F22
        n22 = R23 @ n33 + np.cross(self.Pc22, F22) + np.cross(P23, R23 @ f33) + N22

        f11 = R12 @ f22 + F11
        n11 = R12 @ n22 + np.cross(P12, R12 @ f22) + N11

        Tau = [n11[2], n22[2], -n44[2], -n66[2], -n77[2], -n88[2]]

        return P08, v08, a08, w08, alpha08, Tau



if __name__ == '__main__':
    arm_dyn = Dynamics('ur5e')

    qs = np.array([0, 0, 0, 0, 0, 0])
    qf = np.array([pi/2, pi/3, pi/6, -pi/4, pi/2, pi/1])

    q = qf - qs


    Tau = []

    P = []
    V = []
    A = []
    W = []
    Alpha = []

    TH = []
    dTH = []
    ddTH = []

    total_time = 10
    time = np.linspace(0, total_time, 100)
    for t in time:
        if t<= total_time:
            th = qs + q*((t/total_time) - (1/(2*pi))*sin(2*pi*t/total_time))
            dth = (q/total_time)*(1 - cos(2*pi*t/total_time))
            ddth = ((2*pi*q)/(total_time**2))*sin(2*pi*t/total_time)
        else:
            th = qf
            dth = [0, 0, 0, 0, 0, 0]
            ddth = [0, 0, 0, 0, 0, 0]

        TH.append(th)
        dTH.append(dth)
        ddTH.append(ddth)

        p, v, a, w, alpha, taus = arm_dyn.inverse(th, dth, ddth)
        Tau.append(taus)
        P.append(p)
        V.append(v)
        A.append(a)
        W.append(w)
        Alpha.append(alpha)

    plt.plot(time, Tau)
    plt.show()