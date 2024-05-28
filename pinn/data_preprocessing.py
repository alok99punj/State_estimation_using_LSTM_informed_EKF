import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Results_PINN/Dataset/data.csv")

### calculating the lateral acceleration and longitudinal acceleration from the dataset available 

log_vel = df["xdot"].to_numpy()
lat_vel = df["ydot"].to_numpy()

log_acc = []
lat_acc = []
initial_log = 0.0
initial_lat = 0.0
log_acc.append(initial_log)
lat_acc.append(initial_lat)

for i in range(1,len(log_vel)):
    log_at_instance = (log_vel[i] - log_vel[i-1]) /0.01
    log_acc.append(log_at_instance)
    lat_at_instance = (lat_vel[i] - lat_vel[i-1]) / 0.01 
    lat_acc.append(lat_at_instance)
#### adding the lateral and longitudinal accelerations to the dataframe 
log_acc = np.array(log_acc)
lat_acc = np.array(lat_acc)

df["ax"] = log_acc
df["ay"] = lat_acc 


print(df.head())


#### preprocessing the data for training 

df.replace(" ",np.nan,inplace = True )
df.dropna(inplace = True)
df.to_csv("/Users/alokpunjbagrodia/Desktop/Results_PINN/Dataset/data_values.csv")
#### Normalizing the data and storing it in a new dataframe 
df2 = pd.DataFrame({})
df2["ss"] = df["sideslip_angle_ground_truth"]
df1 = pd.DataFrame({}) ### creating a new dataframe 
df2.replace(" ",np.nan, inplace = True)
df2.dropna(inplace = True)

df1["psi"] = df["psi"] /df["psi"].max()
df1["r"] = df["r"] / df["r"].max()
df1["xdot"] = df["xdot"] / df["xdot"].max()
df1["ydot"] = df["ydot"] / df["ydot"].max()
df1["fyr"] = df["fyr"] / df["fyr"].max()
df1["fyf"] = df["fyf"] / df["fyf"].max()
df1["ax"] = df["ax"] / df["ax"].max()
df1["ay"] = df["ay"] / df["ay"].max()

#### normalized the feature sets to be passed as input 

print(df1.head())

#df1.to_csv("/Users/alokpunjbagrodia/Desktop/Merge/normalized_input.csv")
#df2.to_csv("/Users/alokpunjbagrodia/Desktop/Merge/_output.csv")
 
