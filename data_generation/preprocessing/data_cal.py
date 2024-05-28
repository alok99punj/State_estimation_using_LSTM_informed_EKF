import pandas as pd 
import numpy as np 
import math 
import matplotlib.pyplot as plt
df = pd.read_csv("/Users/alokpunjbagrodia/Desktop/3DOF_Dual_Track_data/test_data.csv")
#print(df.head())
#print(df.info())
### calculation of the front sideslip angle alpha f
df["sideslip_f"] = ((np.arctan(df["ydot"] + 1.4*(df["r"]))) / df["xdot"]) - df["wheel_angle"]
df["sideslip_r"] = ( np.arctan(df["ydot"] + 1.6*(df["r"]))) / df["xdot"]
df["fyft"] = - 12e3 * df["sideslip_f"] * 0.7 * df["fzf"] / 5000
df["fyrt"] = 11e3 * df["sideslip_r"] * 0.7 * df["fzr"] / 5000
df["fyf"] = df["fyft"] * np.cos(df["sideslip_f"])
df["fyr"] = df["fyrt"] * np.cos(df["sideslip_r"])
df["beta"] = np.arctan(df["ydot"] / df["xdot"])
print(df.head())
df.to_csv("/Users/alokpunjbagrodia/Desktop/3DOF_Dual_Track_data/test_data_.csv")
print(min(df["beta"]))


#### plotting the yaw rate and the beta values 

plt.plot(df["r"], color ="red")
plt.plot(df["beta"],color = "blue")
plt.xlabel('Yaw rate')
plt.ylabel('Sideslip angle')
plt.title('Plot1')
plt.grid(True)
plt.show()