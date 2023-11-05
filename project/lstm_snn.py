# import numpy as np
# import torch
# import torch.nn as nn


# class LSTM_SNN(nn.Module):
#     def __init__(self,input_dim,hidden_dim,num_layers,output_dim,device="cpu"):
#         super(LSTM_SNN,self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.output_dim = output_dim
#         self.device = device

#         self.lstm = nn.LSTM(
#             input_size = self.input_dim,
#             hidden_size = self.hidden_dim,
#             num_layers = self.num_layers,
#             batch_first = True,
#             dropout = 0.2
#         )

#     def forward(self,x):

#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)

#         x,(h_t,c_t) = self.lstm(x)

#         return x[:,-1,:]


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Generate some example time series data
np.random.seed(0)
n_samples = 500
time = np.linspace(0, 8, n_samples)
y = np.sin(time)
y_noisy = y + 0.3*np.random.normal(size=n_samples)

plt.plot(time,y_noisy)
plt.show()

# Apply some simple feature extraction (taking the last 10 points as features)
n_features = 10
X = np.zeros((n_samples - n_features, n_features))
for i in range(n_samples - n_features):
    X[i] = y_noisy[i:i+n_features]

# Use MinMaxScaler to scale data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Apply RBM for feature extraction
rbm = BernoulliRBM(n_components=2)
X_transformed = rbm.fit_transform(X_scaled)

# Reshape the data for LSTM
X_transformed = X_transformed.reshape((X_transformed.shape[0], X_transformed.shape[1], 1))

# Split into train and test sets
n_train = 400
X_train, X_test = X_transformed[:n_train], X_transformed[n_train:]
y_train, y_test = y[n_features:n_train+n_features], y[n_train+n_features:]
