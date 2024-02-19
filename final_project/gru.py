import torch
from torch import nn
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler,MinMaxScaler
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





class GRU_module(nn.Module):
	def __init__(self,input_size=10,hidden_size=500,drop=0.1,device="cpu"):
		super(GRU_module, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.device = device
		self.drop=drop
		self.gru1 = nn.GRU(self.input_size, self.hidden_size,num_layers=1,batch_first=True).to(device)
		self.dropout = nn.Dropout(drop)


	def forward(self,x):
		h_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=torch.float32, requires_grad=True).to(self.device)
		out,h_n = self.gru1(x, h_t)
		out = self.dropout(out)
		return out[:,-1,:]
		# return out,h_n,h_n

class GRU(nn.Module):
	def __init__(self,input_size,visible_size,hidden_size,optimizer,criterion,scheduler,epoch,clipping,k,learning_rate_lstm,learning_rate_gbrbm,cd_step,drop,device="cpu"):
		super(GRU, self).__init__()
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
		self.use_scheduler = False
		self.drop = drop

		# Setting lstm layer and Gaussian Binary Restricted Boltzmann Machine
		self.gru_layer = GRU_module(
			input_size=self.input_size,
			hidden_size=self.visible_size,
			drop=drop,
			device=self.device
			)
		self.linear1 = nn.Linear(visible_size,1,device=self.device)

		
		#setting optimizer and learning_rate scheduler
		self.optimizer_gru = self.get_optimizer(optimizer,"lstm")
		if self.use_scheduler:
			# self.scheduler_lstm = self.get_scheduler(scheduler,"lstm")
			self.scheduler_gbrbm = self.get_scheduler(scheduler,"gbrbm")

		#Setting informazion variables
		self.loss = []
		self.lr_list_lstm = []
		self.loss_valid = []

	def eval(self):
		self.gru_layer.eval()
		return

	def get_optimizer(self,optimizer_name,type):
		if optimizer_name=="adam":
			if type == "lstm":
				return torch.optim.Adam(self.gru_layer.parameters(), lr=self.learning_rate_lstm,weight_decay=0.2)
			else:
				return torch.optim.Adam(self.gbrbm.parameters(), lr=self.learning_rate_gbrbm)
		else:
			if type == "lstm":
				return torch.optim.SGD(self.gru_layer.parameters(), lr=self.learning_rate_lstm,weight_decay=0.2)
			else:
				return torch.optim.SGD(self.gbrbm.parameters(), lr=self.learning_rate_gbrbm)

	def get_scheduler(self,scheduler_name,type):
		if scheduler_name == "cosine_anneling":
			if type == "lstm":
				return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_lstm, self.epoch)
			else:
				return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_gbrbm, self.epoch)
		else:
			if type == "lstm":
				return torch.optim.lr_scheduler.ExponentialLR(self.optimizer_lstm, gamma=0.1)
			else:
				return torch.optim.lr_scheduler.ExponentialLR(self.optimizer_gbrbm, gamma=0.1)

	def train(self,train_loader,validation_loader=None):
		for epoch in tqdm(range(self.epoch)):
			print("Current epoch :{}".format(epoch),end="\r")
			loss ,loss_valid = self.train_current_epoch(train_loader,validation_loader)
			# return self.train_current_epoch(train_loader)
			self.loss.append(loss.item())
			self.loss_valid.append(loss_valid)
			if self.use_scheduler:
				self.lr_list_lstm.append(self.optimizer_lstm.param_groups[0]["lr"])
				# self.scheduler_lstm.step()
				self.scheduler_gbrbm.step()
			print("Current epoch :{} , current error: {}".format(epoch,loss),end="\r")
			print("")

	def predict(self,data):
		data_gru = self.lstm_layer(data)
		pred = self.linear1(data_gru)
		return pred.detach().numpy()

	def forward(self,data):
		data_gru = self.gru_layer(data)
		pred = self.linear1(data_gru)
		return pred

	def train_current_epoch(self,train_loader,validation_loader=None):
		self.gru_layer.train()

		loss_vet = []
		loss_grb_vet = []

		for ii, (data,target)  in enumerate(train_loader):
			self.optimizer_gru.zero_grad()

			#--------------------------
			data = data.to(self.device)
			target = target.to(self.device)

			pred = self.forward(data)
			# pred = pred[None,:]
			linear_loss = self.criterion(pred,target)
			linear_loss.backward()
			self.optimizer_gru.step()


			loss_vet.append(linear_loss.item())



		validation_error = []
		if validation_loader != None:
			self.gru_layer.eval()
			with torch.no_grad():
				for ii, (data,target)  in enumerate(validation_loader):
					data = data.to(self.device)
					target = target.to(self.device)

					pred = self.forward(data)
					# pred = pred[None,:]
					linear_loss = self.criterion(pred,target)
					validation_error.append(linear_loss.item())
					

		validation_error = np.array(validation_error).mean()
		return [np.array(loss_vet).mean(),validation_error]