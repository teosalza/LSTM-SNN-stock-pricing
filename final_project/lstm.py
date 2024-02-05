import torch
from torch import nn
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchviz import make_dot, make_dot_from_trace
import torch.nn.functional as F
from torch.autograd import grad
import pandas as pd
import os
import math
from utils import create_window_dataset,split_train_test,split_train_test_valid



class LSTM_module(nn.Module):
	def __init__(self,input_size=10,hidden_size=500,device="cpu"):
		super(LSTM_module, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.device = device
		self.lstm1 = nn.LSTM(self.input_size, self.hidden_size,batch_first=True).to(device)
		self.drop1 = nn.Dropout(0.2)

	def forward(self,x):
		h_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=torch.float32, requires_grad=True).to(self.device)
		c_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=torch.float32, requires_grad=True).to(self.device)
		out,(h_n,c_n) = self.lstm1(x, (h_t, c_t))
		out = self.drop1(out)

		return out[:,-1,:]
		# return out,h_n,h_n

class LSTM(nn.Module):
	def __init__(self,input_size,visible_size,hidden_size,optimizer,criterion,scheduler,epoch,clipping,k,learning_rate_lstm,learning_rate_gbrbm,cd_step,device="cpu"):
		super(LSTM, self).__init__()
		self.input_size = input_size
		self.visible_size = visible_size
		self.hidden_size = hidden_size
		self.criterion = criterion
		self.epoch = epoch
		self.clipping = clipping
		self.k = k
		self.learning_rate_lstm = learning_rate_lstm
		self.learning_rate_gbrbm = learning_rate_gbrbm
		self.cd_step = cd_step
		self.device=device
		self.dot = ""
		self.use_scheduler = True

		# Setting lstm layer and Gaussian Binary Restricted Boltzmann Machine
		self.lstm_layer = LSTM_module(
			input_size=self.input_size,
			hidden_size=self.visible_size,
			device=self.device
			)
		
		self.linear1 = nn.Linear(visible_size,1,device=self.device)

		#setting optimizer and learning_rate scheduler
		self.optimizer_lstm = self.get_optimizer(optimizer,"lstm")

		#Setting informazion variables
		self.loss = []
		self.loss_gbrbm = []
		self.lr_list_lstm = []
		self.lr_list_gbrbm = []
		self.loss_valid = []

	def eval(self):
		self.lstm_layer.eval()
		return

	def get_optimizer(self,optimizer_name,type):
		if optimizer_name=="adam":
			return torch.optim.Adam(self.lstm_layer.parameters(), lr=self.learning_rate_lstm,weight_decay=0.2)
		else:
			return torch.optim.SGD(self.lstm_layer.parameters(), lr=self.learning_rate_lstm,weight_decay=0.2)

	def train(self,train_loader,validation_loader=None):
		for epoch in tqdm(range(self.epoch)):
			print("Current epoch :{}".format(epoch),end="\r")
			loss,loss_valid = self.train_current_epoch(train_loader,validation_loader)
			# return self.train_current_epoch(train_loader)
			self.loss.append(loss)
			self.loss_valid.append(loss_valid)
			print("Current epoch :{} , current error: {}".format(epoch,loss),end="\r")
			print("")

	def predict(self,data):
		data_lstm = self.lstm_layer(data)
		# pred = self.linear2(self.linear1(data))
		pred = self.linear1(data)
		return pred.detach().numpy()

	def forward(self,data):
		data_lstm = self.lstm_layer(data)
		# pred = self.linear2(self.linear1(data_lstm))
		pred = self.linear1(data_lstm)
		return pred

	def train_current_epoch(self,train_loader,validation_loader=None):
		self.lstm_layer.train()

		for ii, (data,target)  in enumerate(train_loader):
			self.optimizer_lstm.zero_grad()

			#--------------------------
			data = data.to(self.device)
			target = target.to(self.device)

			pred = self.forward(data)
			linear_loss = self.criterion(pred,target)
			linear_loss.backward()

			# pred = self.gbrbm.compute_linear_layer(data_lstm)
			# linear_loss = self.criterion(pred,target)


			# make_dot(pred.mean(), params=dict(self.named_parameters()))

			# grads_lstm = grad(linear_loss,data_lstm)


			# linear_loss.backward()

			self.optimizer_lstm.step()

		validation_error = []
		if validation_loader != None:
			self.lstm_layer.eval()
			with torch.no_grad():
				for ii, (data,target)  in enumerate(validation_loader):
					data = data.to(self.device)
					target = target.to(self.device)

					pred = self.forward(data)
					linear_loss = self.criterion(pred,target)
					validation_error.append(linear_loss.item())
					

		validation_error = np.array(validation_error).mean()
		return [linear_loss,validation_error]
