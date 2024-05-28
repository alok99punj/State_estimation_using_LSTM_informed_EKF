##### Note: Monte Carlo runs are executed to calculate the mean error plot 

import numpy as np 
import math 
import pandas as pd 
import matplotlib.pyplot as plt 
from PINN_ekf import KalmanFilter
from white_gaussian import white_gaussian
import random
from lstm import LSTMModel 
import torch
df = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/all_sets/final1_set.csv")
#df = pd.read_csv("/Users/alokpunjbagrodia/Desktop/3DOF_Dual_Track_data/merged_1.csv")
#df = pd.read_csv("/Users/alokpunjbagrodia/Desktop/3DOF_Dual_Track_data/test_data_.csv")
#df = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/all_sets/testing_set.csv")
model = LSTMModel(input_size = 5, hidden_size = 64 , output_size = 1)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
selected_columns = df[["vx","vy","psi","r","wheel_angle"]]
x = selected_columns.values #### x is the input to the lstm 
y = df["beta"].values   #### y is the output of the lstm 
##### converting x and y to tensor 
x = torch.tensor(x,dtype = torch.float32)
y = torch.tensor(y,dtype = torch.float32)
vy = df["vy"].to_numpy()
print(vy.shape)
vx = df["vx"].to_numpy()
d = df["wheel_angle"].to_numpy()
fyf = df["fyf"].to_numpy()
fyr = df["fyr"].to_numpy()
r = df["r"].to_numpy()
beta = df["beta"].to_numpy()
##### initializing initial values #####
x0 = np.array([0,0])
x0 = x0.reshape(2,1)
P0 = 1 * np.eye(2,2)
#Q =   white_gaussian(9) * np.eye(2,2)
Q = np.diag(np.full(2,white_gaussian(1)))
#R =  white_gaussian(7) * np.eye(5,5)
R = np.diag(np.full(5,white_gaussian(1)))
dt = 0.01
I = np.array([[1,0],[0,1]])
ekf = KalmanFilter(x0,P0,Q,R,dt,I) ### object instantiation
predicted = []
error =[]
total = 0 
runs = []
total = [0] * (len(vx))
error = [0] * (len(vy))
mse = 0 
mean_beta = np.mean(beta)
me = 0 
mse_array = []
me_array=[]
#### estimation of beta using EKF 
runs = 1
mse_runs = 0 
me_runs = 0 
for c in range(runs):
    #x0 = np.array([white_gaussian(12),white_gaussian(10)])
    x0 = np.array([0,0])
    x0 = x0.reshape(2,1)
    P0 = 1 * np.eye(2,2)
    random.seed(3)
    np.random.seed(3)
    Q = np.diag(np.full(2,white_gaussian(5)))
    R =  np.diag(np.full(5,white_gaussian(5)))
    for i in range(0,len(beta)-1):
        j = i+1 #### running variable for timeslot K 
        #### loop it for all data values to calculate estimate over the entire dataset 
        #### calculation of data values required for process model by using the current states 
        ##### lstm output for timestep j 
        output_lstm_beta = model(x[i])
        sideslip_front = np.arctan(x0[0][0] + (1.4 * x0[1][0]) / vx[i]) - d[i]
        sideslip_rear = np.arctan(x0[0][0] - (1.6 * x0[1][0]) / vx[i])
        ff = 12e3 * sideslip_front
        fr = 11e3 * sideslip_rear
        dataset_values = [ff,fr,d[i],vx[i],x0[1][0]] 
        ### X_k_apriori for timeslot k 
        x_k_apriori = ekf.a_priori_state_k(x0,dataset_values)


        #### calculating the jacobian 

        A_k_minus_1, Ck =  ekf.jacobian(vx[i],x_k_apriori[1][0],vx[j])
        #print(Ck.shape)
        #### Calculating the Apriori Covariance Matrix 

        P_apriori = ekf.a_priori_P(A_k_minus_1,P0)
        #print(P_apriori.shape)
        # #### Calculating the Kalman Gain 

        K = ekf.kalman_gain(P_apriori,Ck)
        #print(K.shape)
        # #### State Updation for timeslot k 
        # ##### Calculation of the observation vector for timeslot k
        ay = 1/2000 * (fyf[j] * math.cos(d[j]) + fyr[j])
        observation_vector = np.array([ay,r[j],fyf[j],fyr[j],beta[j]])



        ### reshaping the vectors for vector multiplication 

        observation_vector = observation_vector.reshape(5,1)
        Ck = Ck.reshape(5,2)
        x_k_apriori = x_k_apriori.reshape(2,1)
        x_a_posteriori, P_a_posteriori = ekf.state_updation(x_k_apriori, K,observation_vector,Ck,P_apriori,output_lstm_beta.item())
        #### updating the state and covariance matrix , Treated as K-1 aposteriori matrix(K-1) for the next timeslot 
        #print(x_a_posteriori.shape)
        x0 = x_a_posteriori
        P0 = P_a_posteriori
        x0 = x0.reshape(2,1)
        beta_pred = np.arctan(x_a_posteriori[0]/vx[j])
        predicted.append(beta_pred)
        error[i] = ((beta[j] - beta_pred))
        #error[i] = abs(vy[i] - x0[0])
        total[i] = total[i] + error[i][0]
        #total[i] = total[i] + error[i][0]
        mse = mse + (beta_pred - mean_beta)**2
        me = me + abs(beta_pred - mean_beta)
    mse_runs = mse_runs + (mse/(len(beta)))
    me_runs = me_runs + (me/(len(beta)))
    me = 0 
    mse = 0 
mse_runs = mse_runs / runs 
me_runs = me_runs / runs 
mse_runs = mse_runs * (180/math.pi)
me_runs = me_runs * (180/ math.pi)

print(mse_runs)
print(me_runs)


predicted = np.array(predicted)
### saving the predicted beta array


# print(predicted.shape)


plt.plot(beta,label = "Observed Sideslip Angle",color = "red")
plt.plot(predicted,label ="Estimated Sideslip Angle",color = "green")
plt.xlabel("Data Points")
plt.ylabel("Sideslip Angle (Rad)")
plt.legend()
plt.grid(True)
plt.show()

# plt.plot(monte_carlo_runs[1:15000],label= "Mean Error",color = "green")
# plt.ylabel("Mean Error")
# plt.legend()
# plt.grid(True)
# plt.show()


# plt.plot(mse_array,label = "mse",color = "magenta")
# plt.grid(True)
# plt.show()
### saving the numpy array 
#np.save("monte_carlo_run_ekf.npy",monte_carlo_runs)

#np.save("ekf.npy",predicted)

