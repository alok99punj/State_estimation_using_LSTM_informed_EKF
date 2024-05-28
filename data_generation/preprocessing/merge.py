import pandas as pd
####### psi file ########
# Read the CSV file into a DataFrame
df = pd.read_csv("/Users/alokpunjbagrodia/Desktop/vehicle_model_simulink/fzr.csv")

# Transpose the DataFrame
transposed_df = df.T
#print(transposed_df.head(3000))
# Save the transposed DataFrame to a new CSV file
#transposed_df.to_csv('/Users/alokpunjbagrodia/Desktop/Vehicle 3DOF model Dataset /vehicle moving on straight roda with a spped of 30 km:hr/transposed_file.csv', header=False)


# Drop the first two columns
df = transposed_df.drop(transposed_df.columns[:1], axis=1)

df.rename(columns={1: 'fzr'}, inplace=True)
##### when finding the mean of certain values 
# mean_values = df[[1, 2]].mean(axis=1)
# df["fzr"] = mean_values

#### keeping certain columns 

# Drop columns not in the list
# columns_to_keep = ["fzf"]
# df = df.drop(columns=df.columns.difference(columns_to_keep))
# print(df.head())




# #psi = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Vehicle 3DOF model Dataset /vehicle moving on straight roda with a spped of 30 km:hr/psi1.csv")
df.to_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/fzr.csv")
print(df.head())








