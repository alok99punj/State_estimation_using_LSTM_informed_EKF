import pandas as pd



# Read each CSV file into separate DataFrames
df1 = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/fzf.csv")
df2 = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/fzr.csv")
df3 = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/steer.csv")
df4 = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/r.csv")
df5 = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/psi.csv")
df6 = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/wheel_angle.csv")
df7 = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/xdot.csv")
df8 = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/ydot.csv")
# Concatenate the DataFrames column-wise
merged_df = pd.concat([df1, df2, df3,df4,df5,df6,df7,df8], axis=1)
#sliced_df = merged_df.iloc[:1501]

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('/Users/alokpunjbagrodia/Desktop/3DOF_Dual_Track_data/test_data.csv', index=False)
