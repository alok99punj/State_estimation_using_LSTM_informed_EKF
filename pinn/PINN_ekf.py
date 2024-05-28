import math 
import numpy as np 
import pandas as pd 
class KalmanFilter:
    def __init__(self,x0,P0,Q,R,dt,I):
        """_summary_

        Args:
            x0 (_type_): x0 is the initial state at k-1 which is given a random value 
            P0 (_type_): P0 is the initial covariance matrix which is given a initial random value 
            Q (_type_): Q is the process noise given an initial value of 0.0001
            R (_type_): R is the initial observation noise given a value of 0.00001
        """
        self.x0 = x0 
        self.P0 = P0
        self.Q = Q
        self.R = R
        self.dt = dt
        self.I = I
    def non_linear_process_model(self,dataset_values):
        """
        The process model is defined using the non linear bicycle model 
        the following variables with constant values are considered while constructing the process model
        
        lf = 1.4 
        lr = 1.6 
        m = 2000
        Fznom = 5000 
        I = 4000 
        cf = 12e3 
        cr = 11e3
        friction coefficient = 0.7 
        
        fyf, fyr , d , vx and r are dataset values for k-1 instant 
        """
        fyf, fyr, d,vx,r = dataset_values
        vy_dot = 1/2000 * ((fyf*math.cos(d)) + fyr) - (vx * r)
        r_dot = 1/4000 * ((fyf*math.cos(d)*1.4) - (1.6*fyr))
        return vy_dot, r_dot  
    def a_priori_state_k(self,x_k_minus_1,dataset_values):
        """_summary_

        Args:
            x_k_minus_1 (_type_): it is the state estimate(a posteriori) for k-1 
            a priori x is calculated for timeslot k 
        """
        vy_dot, r_dot = self.non_linear_process_model(dataset_values)
        dot = np.array([vy_dot,r_dot]) 
        dot =   dot * self.dt
        dot = dot.reshape(2,1)
        x_k_apriori = dot 
        x_k_apriori = x_k_apriori.reshape(2,1) ### reshaping the output 
        return x_k_apriori 
    def jacobian(self,vx1,vy,vx2):
        """_summary_

        Args:
            vx (_type_): vx is from timeslot k 
            vx1 denotes the vx at timeslot k-1 
            vx2 denotes the vx at timeslot k s
            vy is the apriori value from the state space 
        """
        a = -vx1 * self.dt
        b = vx2/(vx2**2 + vy**2)
        A_k_minus_1 = np.array([[1,a],[0,1]])
        Ck = np.array([[0,0,0,0,b],[0,1,0,0,0]])
        
        return A_k_minus_1,Ck
    
    def a_priori_P(self,A_k_minus_1,P_k_minus_1):
        """_summary_

        Args:
            A_k_minus_1 (_type_): Jacobian of process model 
            P_k_minus_1 (_type_): Covariance matrix K-1 
            P_apriori : is P_minus_k

        Returns:
            _type_: _description_
        """
        P_apriori = (A_k_minus_1 @  P_k_minus_1 @ A_k_minus_1.T) + self.Q
        
        return  P_apriori 
    def kalman_gain(self,P_apriori,Ck):
        inter = (Ck.T @ P_apriori @ Ck) + self.R
        K = P_apriori @ Ck @ np.linalg.inv(inter)
        return K 
    def state_updation(self,x_k_apriori,K,observation_vector,Ck,P_apriori,beta):
        gk = Ck @ x_k_apriori
        #gk[4][0] = beta
        observation_vector[4][0] = beta 
        inter = self.I - (K @ Ck)
        error = observation_vector - gk 
        x_a_posteriori = x_k_apriori + (K @ error)
        P_a_posteriori = (self.I - (K @ Ck)) @ P_apriori @ (inter.T) + (K @ self.R @ K.T)
        
        return x_a_posteriori, P_a_posteriori
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        