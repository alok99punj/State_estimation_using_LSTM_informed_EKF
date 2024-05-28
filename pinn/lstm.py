import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        fc1_out = self.tanh(self.fc1(lstm_out[-1]))
        output = self.fc2(fc1_out)
        return output
# Define input and output sizes
input_size = 5
output_size = 1
hidden_size = 64

# Instantiate the model
model = LSTMModel(input_size, hidden_size, output_size)
print(model)
