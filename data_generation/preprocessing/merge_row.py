import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
# df1 = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/all_sets/set1.csv")
# df2 = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/all_sets/set2.csv")
# df3 = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/all_sets/set3.csv")
df4 = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/all_sets/set4.csv")
df5 = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/all_sets/set5.csv")
df6 = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/all_sets/set6.csv")
final_df = pd.concat([df4,df5,df6],axis=0,ignore_index= True)
#final_df.to_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/all_sets/merged_set.csv")
required_columns = ["fzf","fzr","steer_angle","vx","vy","r","psi","wheel_angle","sideslip_f","sideslip_r","fyft","fyft","fyf","fyr","beta"]
filtered_columns = final_df[required_columns]
print(filtered_columns.head())
filtered_columns.to_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/all_sets/testing_set.csv")



