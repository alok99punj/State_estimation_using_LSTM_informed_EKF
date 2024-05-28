import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn 
from sklearn.model_selection import train_test_split
from lstm import LSTMModel 

# Load data
data = pd.read_csv("/Users/alokpunjbagrodia/Desktop/Bicycle-Model-Data/all_sets/final1_set.csv")
df_train = data[["vx", "vy", "psi", "r", "wheel_angle"]]
x = df_train.values
y = data["beta"].values

# Split data into train and test sets 
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors 
x_train = torch.tensor(x_train, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Adding an extra dimension for compatibility with model
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Instantiate the LSTM model 
model = LSTMModel(input_size=5, hidden_size=64, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 50
training_loss = []

for epoch in range(num_epochs):
    epoch_loss = 0 
    for i in range(len(x_train)):
        optimizer.zero_grad()
        output = model(x_train[i:i+1])  # Forward pass for each instance
        loss = criterion(output, y_train[i:i+1])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    training_loss.append(epoch_loss / len(x_train))
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss / len(x_train)}')

# Validation
model.eval()
with torch.no_grad():
    val_outputs = model(x_val)
    val_loss = criterion(val_outputs, y_val)
    print(f'Validation Loss: {val_loss.item()}')

# Save model weights
torch.save(model.state_dict(), 'model_weights.pth')
print("Model weights saved successfully.")
