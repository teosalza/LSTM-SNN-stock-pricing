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
from sklearn.model_selection import KFold
import math
from utils import create_window_dataset,split_train_test,split_train_test_valid
from sklearn.decomposition import PCA





class GBRBM(torch.nn.Module):
	def __init__(self,visible_size,hidden_size,init_var=1e-0,cd_step=1,cd_burning=0,device="cpu"):
		super().__init__()
		self.cd_step = cd_step
		self.cd_burning = cd_burning
		self.visible_size = visible_size
		self.hidden_size = hidden_size
		self.init_var = init_var
		self.device=device

		
		self.linear_layer = nn.Linear(hidden_size,1).to(self.device)
		self.dropout = nn.Dropout(0.1)
		self.W = nn.Parameter(torch.Tensor(visible_size, hidden_size).to(self.device))
		self.b = nn.Parameter(torch.Tensor(hidden_size).to(self.device))
		self.mu = nn.Parameter(torch.Tensor(visible_size).to(self.device))
		self.log_var = nn.Parameter(torch.Tensor(visible_size).to(self.device))
		self.reset_parameters()

	def reset_parameters(self):
		nn.init.normal_(self.W,
                        std=1.0 * self.init_var /
                        np.sqrt(self.visible_size + self.hidden_size))
		nn.init.constant_(self.b, 0.0)
		nn.init.constant_(self.mu, 0.0)
		nn.init.constant_(self.log_var,
                          np.log(self.init_var))  # init variance = 1.0

	def get_var(self):
		return self.log_var.exp().clip(min=1e-8)

	@torch.no_grad()
	def CD_grad(self,v):
		# v = v.view(v.shape[0], -1)
		prob_h, h, pos_eng = self.positive_gradient(v)

		# negative gradient
		v_neg = torch.randn_like(v)
		neg_eng = self.negative_grad(v_neg)

		return [pos_eng,neg_eng]
		for name, param in self.named_parameters():
			param.grad = pos_eng[name] - neg_eng[name]

	@torch.no_grad()
	def negative_grad(self, v):
		var = self.get_var()
		var_mean = var.mean().item()
		
		#Normal Gibb approximation
		samples,_ = self.Gibbs_sampling_vh(v,
										num_steps=self.cd_step,
										burn_in=self.cd_burning)
		v_neg = torch.cat([xx[0] for xx in samples], dim=0)
		h_neg = torch.cat([xx[1] for xx in samples], dim=0)
		grad = self.energy_grad_param(v_neg, h_neg)
		return grad

	@torch.no_grad()
	def Gibbs_sampling_vh(self, v, num_steps=10, burn_in=0):
		samples, var = [], self.get_var()
		prob_h_samples = []
		std = var.sqrt()
		h = torch.bernoulli(self.prob_h_given_v(v, var))
		for ii in range(num_steps):
            # backward sampling
			mu = self.prob_v_given_h(h)
			v = mu + torch.randn_like(mu) * std

            # forward sampling
			prob_h = self.prob_h_given_v(v, var)
			h = torch.bernoulli(prob_h)
			if ii >= burn_in:
				samples += [(v, h)]
				prob_h_samples.append(prob_h)
		return samples,prob_h_samples

	# @torch.no_grad()
	def prob_h_given_v(self, v, var):
		return torch.sigmoid((v / var).mm(self.W) + self.b)

	@torch.no_grad()
	def prob_v_given_h(self, h):
		return h.mm(self.W.T) + self.mu

	@torch.no_grad()
	def positive_gradient(self, v):
		prob_h = self.prob_h_given_v(v, self.get_var())
		h = torch.bernoulli(prob_h)
		pos_energy = self.energy_grad_param(v,h)
		return prob_h,h,pos_energy

	@torch.no_grad()
	def energy_grad_param(self, v, h):
        # compute the gradient (parameter) of energy averaged over batch size
		var = self.get_var()
		grad = {}
		grad['W'] = -torch.einsum("bi,bj->ij", v / var, h) / v.shape[0]
		grad['b'] = -h.mean(dim=0)
		grad['mu'] = ((self.mu - v) / var).mean(dim=0)
		grad['log_var'] = (-0.5 * (v - self.mu)**2 / var +
		                   ((v / var) * h.mm(self.W.T))).mean(dim=0)
		
		# grad["log_var"] = torch.zeros((500)).to(self.device)
		return grad

	@torch.no_grad()
	def reconstruction(self, v):
		v, var = v.view(v.shape[0], -1), self.get_var()
		prob_h = self.prob_h_given_v(v, var)
		v_bar = self.prob_v_given_h(prob_h)
		return F.mse_loss(v, v_bar)

	def compute_linear_layer(self,data):
		v = data
		# samples,prob_h = self.Gibbs_sampling_vh(data,1,0)
		samples, var = [], self.get_var()
		prob_h_samples = []
		std = var.sqrt()
		h = torch.bernoulli(self.prob_h_given_v(v, var).requires_grad_()).requires_grad_()
            # backward sampling
		mu = self.prob_v_given_h(h)
		v = mu + torch.randn_like(mu) * std

        # forward sampling
		prob_h = self.prob_h_given_v(v, var)
		# return self.relu(self.linear_layer(prob_h))
		return self.linear_layer(self.dropout(prob_h))
		# return self.linear_layer1(self.linear_layer(prob_h))

	def forward(self,data):
		# samples,prob_h = self.Gibbs_sampling_vh(data,1,0)
		samples, var = [], self.get_var()
		prob_h_samples = []
		std = var.sqrt()
		h = torch.bernoulli(self.prob_h_given_v(data, var).requires_grad_()).requires_grad_()
            # backward sampling
		mu = self.prob_v_given_h(h).requires_grad_()
		v = (mu + torch.randn_like(mu) * std).requires_grad_()

        # forward sampling
		prob_h = self.prob_h_given_v(data, var).requires_grad_()
		# return self.relu(self.linear_layer(prob_h))
		return self.linear_layer(self.dropout(prob_h))
		# return self.linear_layer1(self.linear_layer(prob_h))

class LSTM_module(nn.Module):
	def __init__(self,input_size=10,hidden_size=500,device="cpu"):
		super(LSTM_module, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.device = device
		self.lstm1 = nn.LSTM(self.input_size, self.hidden_size,batch_first=True).to(device)
		self.dropout = nn.Dropout(0.1)


	def forward(self,x):
		h_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=torch.float32, requires_grad=True).to(self.device)
		c_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=torch.float32, requires_grad=True).to(self.device)
		out,(h_n,c_n) = self.lstm1(x, (h_t, c_t))
		out = self.dropout(out)
		return out[:,-1,:]
		# return out,h_n,h_n

class LSTM_GBRBM(nn.Module):
	def __init__(self,input_size,visible_size,hidden_size,optimizer,criterion,scheduler,epoch,clipping,k,learning_rate_lstm,learning_rate_gbrbm,cd_step,device="cpu"):
		super(LSTM_GBRBM, self).__init__()
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

		# Setting lstm layer and Gaussian Binary Restricted Boltzmann Machine
		self.lstm_layer = LSTM_module(
			input_size=self.input_size,
			hidden_size=self.visible_size,
			device=self.device
			)

		self.gbrbm = GBRBM(
			visible_size=self.visible_size,
			hidden_size=self.hidden_size,
			cd_step=cd_step,
			device=self.device
			).to(device)

		#setting optimizer and learning_rate scheduler
		self.optimizer_lstm = self.get_optimizer(optimizer,"lstm")
		self.optimizer_gbrbm = self.get_optimizer(optimizer,"gbrbm")
		if self.use_scheduler:
			# self.scheduler_lstm = self.get_scheduler(scheduler,"lstm")
			self.scheduler_gbrbm = self.get_scheduler(scheduler,"gbrbm")

		#Setting informazion variables
		self.loss = []
		self.loss_gbrbm = []
		self.lr_list_lstm = []
		self.lr_list_gbrbm = []
		self.loss_valid = []

	def eval(self):
		self.lstm_layer.eval()
		self.gbrbm.eval()
		return

	def get_optimizer(self,optimizer_name,type):
		if optimizer_name=="adam":
			if type == "lstm":
				return torch.optim.Adam(self.lstm_layer.parameters(), lr=self.learning_rate_lstm,weight_decay=0.2)
			else:
				return torch.optim.Adam(self.gbrbm.parameters(), lr=self.learning_rate_gbrbm)
		else:
			if type == "lstm":
				return torch.optim.SGD(self.lstm_layer.parameters(), lr=self.learning_rate_lstm,weight_decay=0.2)
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
			loss_gbrbm, loss ,loss_valid = self.train_current_epoch(train_loader,validation_loader)
			# return self.train_current_epoch(train_loader)
			self.loss.append(loss.item())
			self.loss_gbrbm.append(loss_gbrbm)
			self.loss_valid.append(loss_valid)
			if self.use_scheduler:
				self.lr_list_lstm.append(self.optimizer_lstm.param_groups[0]["lr"])
				self.lr_list_gbrbm.append(self.optimizer_gbrbm.param_groups[0]["lr"])
				# self.scheduler_lstm.step()
				self.scheduler_gbrbm.step()
			print("Current epoch :{} , current error: {}, current error gbrmb: {}".format(epoch,loss,loss_gbrbm),end="\r")
			print("")

	def predict(self,data):
		data_lstm = self.lstm_layer(data)
		pred = self.gbrbm.compute_linear_layer(data_lstm)
		return pred.detach().numpy()

	def forward(self,data):
		data_lstm = self.lstm_layer(data)
		# data_lstm = self.dropout(data_lstm)
		pred = self.gbrbm(data_lstm)
		return pred

	def train_current_epoch(self,train_loader,validation_loader=None):
		self.lstm_layer.train()
		self.gbrbm.train()
		# x_train = train_loader[0]
		# y_train = train_loader[1]

		loss_vet = []
		loss_grb_vet = []
		kfold =KFold(n_splits=10,shuffle=True)

		for ii, (data,target)  in enumerate(train_loader):
		# for train_index, test_index in kfold.split(x_train, y_train):  
			self.optimizer_lstm.zero_grad()
			self.optimizer_gbrbm.zero_grad()

			#--------------------------
			data = data.to(self.device)
			target = target.to(self.device)

			pred = self.forward(data)
			# pred = pred[None,:]
			linear_loss = self.criterion(pred,target)
			loss_vet.append(linear_loss.item())

			

			# self.dot = make_dot(pred.mean(),params=dict(self.named_parameters()))

			data_lstm = self.lstm_layer(data)

			# example_loss = self.criterion(data_lstm,data_lstm*1.4)
			# example_loss.backward()

			# copy_data = data_lstm.detach().clone()
			[pos_eng,neg_eng] = self.gbrbm.CD_grad(data_lstm)
			if self.clipping > 0:
				nn.utils.clip_grad_norm_(self.gbrbm.parameters(),self.clipping)
			linear_loss.backward()

			# pred = self.gbrbm.compute_linear_layer(data_lstm)
			# linear_loss = self.criterion(pred,target)


			# make_dot(pred.mean(), params=dict(self.named_parameters()))

			# grads_lstm = grad(linear_loss,data_lstm)

			for name, param in self.gbrbm.named_parameters():
				if "layer" not in name:
						param.grad = pos_eng[name] - neg_eng[name]
					# if "log" not in name:
						# param.grad = pos_eng[name] - neg_eng[name]
					

			# linear_loss.backward()

			self.optimizer_lstm.step()
			self.optimizer_gbrbm.step()
			# if ii == len(train_loader) - 1:
			recon_loss = self.gbrbm.reconstruction(data_lstm).item()
			loss_grb_vet.append(recon_loss)

		validation_error = []
		if validation_loader != None:
			self.lstm_layer.eval()
			self.gbrbm.eval()
			with torch.no_grad():
				list_val_y = []
				list_target_y = []
				for ii, (data,target)  in enumerate(validation_loader):
					data = data.to(self.device)
					target = target.to(self.device)

					pred = self.forward(data)
					# pred = pred[None,:]
					linear_loss = self.criterion(pred,target)
					validation_error.append(linear_loss.item())

					for i in range(pred.shape[0]):
						list_val_y.append(pred[i].to("cpu").numpy())
						list_target_y.append(target[i].to("cpu").numpy())
						
					
				# plt.plot(np.arange(len(list_val_y)),list_val_y)
				# plt.plot(np.arange(len(list_val_y)),list_target_y)
				# plt.show()
					
		validation_error = np.array(validation_error).mean()
		return [np.array(loss_grb_vet).mean(),np.array(loss_vet).mean(),validation_error]
		# return [recon_loss,linear_loss,validation_error]

