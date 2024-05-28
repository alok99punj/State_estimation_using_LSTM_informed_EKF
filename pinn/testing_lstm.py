import torch 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from lstm import LSTMModel
import math
df = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/all_sets/final1_set.csv")
#df = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/all_sets/final1_set.csv")
#df = pd.read_csv("/Users/alokpunjbagrodia/Desktop/3DOF_Dual_Track_data/test_data_.csv")
#df = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/all_sets/training_set.csv")
#df = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/all_sets/testing_set.csv")
selected_columns = df[["vx","vy","psi","r","wheel_angle"]]
x = selected_columns.values 
y = df["beta"].values
print(y.shape)
y = y.reshape(y.shape[0],1)
beta = df["beta"].to_numpy()
mean_y = np.mean(beta)
print(max(beta),min(beta))
print(mean_y)
model = LSTMModel(input_size = 5, hidden_size = 64, output_size = 1)
model.load_state_dict(torch.load('model_weights.pth'))
#### testing the LSTM
model.eval()
#### converting the array to a tensor 
x = torch.tensor(x,dtype = torch.float32)
y = torch.tensor(y,dtype = torch.float32)
mean_y = torch.tensor(mean_y, dtype = torch.float32)
#### testing the lstm model for a manuever 
a = []
a2 = []
predicted_output = []
labeled= []
mse = 0 
me = 0 
def custom_loss(output, label):
    loss = (label - output)**2 
    return loss
def ME(output,label):
    loss = abs(label - output)
    return loss
for i in range(len(beta)):
    with torch.no_grad():
        output = model(x[i])
        #print(output.shape)
        predicted_output.append(output.item())
        label = mean_y.reshape(1,1)
        labeled.append(y[i].item())
        loss = custom_loss(output,label)
        loss2 = ME(output,label)
        mse = mse + loss.item()
        me = me + loss2.item()
#loaded_ekf_error = np.load("/Users/alokpunjbagrodia/Desktop/ekf/monte_carlo_run_ekf.npy")

#### plotting the error after testing the lstm model 

mse = mse / len(beta)
me = me / len(beta)
mse = mse * (180/math.pi)
me = me * (180/math.pi)
print(f'-> the mse avg is {mse}')
print(f'-> the mae avg is {me}')



# plt.plot(a2,color = "red",label = "LSTM error")
# plt.plot(loaded_ekf_error, color= "blue", label = "ekf error")
# plt.legend()
# plt.xlabel("data points")
# plt.ylabel("error")
# plt.grid(True)
# plt.show()
pinn_predicted = np.load("predicted.npy")
ekf_predicted = np.load("ekf.npy")

 ### plotting the predicted and actual values 

predicted_output = np.array(predicted_output)
y = np.array(y)
ekf_predicted = ekf_predicted * (180/math.pi)
pinn_predicted = pinn_predicted * (180/math.pi)
predicted_output = predicted_output *(180/math.pi)
y =  y * (180/math.pi)
plt.plot(predicted_output,color = "green", label = "LSTM Estimation")
plt.plot(y,color = "red", label = "Observed values")
#plt.plot(ekf_predicted, color = "blue", label ="EKF Estimation")
#plt.plot(pinn_predicted, color = "black",label = "PINN Estimation" )
plt.xlabel("data points")
plt.ylabel("sideslip angle (deg)")
plt.title(" Steering Maneuver Performed ")
plt.legend()
plt.grid(True)
plt.show()
#plt.savefig("overlap.png")

    
    
    
    
    
    
    