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
sys.setrecursionlimit(10000)



class ANN(nn.Module):
	def __init__(self,input_size,hidden_size,optimizer,criterion,epoch,learning_rate,device="cpu"):
		super(ANN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.criterion = criterion
		self.epoch = epoch
		self.learning_rate = learning_rate
		self.device=device
		self.use_scheduler = False

		# Setting lstm layer and Gaussian Binary Restricted Boltzmann Machine
		self.input_layer = nn.Linear(self.input_size,self.hidden_size,device=device) 
		self.hidden_layer = nn.Linear(self.hidden_size,self.hidden_size,device=device) 
		self.dropout = nn.Dropout(0.1)
	
		self.optimizer = self.get_optimizer(optimizer)
		
		self.loss = []
		self.loss_valid = []

	

	def get_optimizer(self,optimizer_name):
		if optimizer_name=="adam":
			return torch.optim.Adam(self.parameters(), lr=self.learning_rate,weight_decay=0.01)
		else:
			return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
			

	def train_model(self,train_loader,validation_loader=None):
		for epoch in tqdm(range(self.epoch)):
			print("Current epoch :{}".format(epoch),end="\r")
			loss,val_loss = self.train_current_epoch(train_loader,validation_loader)
			# return self.train_current_epoch(train_loader)
			self.loss.append(loss)
			self.loss_valid.append(val_loss)
			print("Current epoch :{} , current error: {}".format(epoch,loss),end="\r")
			print("")

	def predict(self,data):
		return self.forward(data)

	def forward(self,data):
		data = data.view(data.shape[0],self.input_size)
		lay1 = self.input_layer(data)
		drop1 = self.dropout(lay1)
		out = self.hidden_layer(drop1)
		return out

	def train_current_epoch(self,train_loader,validation_loader=None):
		self.train()
		loss_vet = []

		for ii, (data,target)  in enumerate(train_loader):
			self.optimizer.zero_grad()

			data = data.to(self.device)
			target = target.to(self.device)

			pred = self.forward(data)
			linear_loss = self.criterion(pred,target)
			linear_loss.backward()
			loss_vet.append(linear_loss.item())

			

			self.optimizer.step()

		validation_error = []
		if validation_loader != None:
			self.eval() 
			with torch.no_grad():
				for ii, (data,target)  in enumerate(validation_loader):
					data = data.to(self.device)
					target = target.to(self.device)

					pred = self.forward(data)
					# pred = pred[None,:]
					linear_loss = self.criterion(pred,target)
					validation_error.append(linear_loss.item())
				# plt.show()
					
		validation_error = np.array(validation_error).mean()
		return [np.array(loss_vet).mean(),validation_error]



if __name__ == "__main__":
	# Instantiate the parser
	parser = argparse.ArgumentParser(description="LSTM-SNN training.", epilog="""-------------------""")

	# Required positional arguments
	parser.add_argument("--dataset-name", type=str,
						help="[string] specify the name.",required=True)
	parser.add_argument("--window-size", type=int,
						help="[int] specify the window size.",required=True)

	args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

	DATASET_NAME = args.dataset_name
	WINDOW_SIZE = args.window_size

	dataset = pd.read_csv(DATASET_NAME)

	#DROP NA Values starting from 26th(25) position
	dataset.dropna(inplace=True)
	dataset.index = dataset["Date"]
	dataset = dataset.iloc[:,1:]

	scaler = StandardScaler()
	# scaled_dataset = scaler.fit_transform(dataset)
	normal = MinMaxScaler(feature_range=(-1, 1))
	scaled_dataset = normal.fit_transform(dataset)

	x_dset,y_dset = create_window_dataset(scaled_dataset,WINDOW_SIZE)

	split_size = 0.8
	valid_size = 0.1
	x_train,y_train,x_valid,y_valid,x_test,y_test = split_train_test_valid(x_dset,y_dset,split_size,valid_size)
	# x_train,y_train,x_test,y_test = split_train_test(x_dset,y_dset,split_size)

	x_train = torch.from_numpy(x_train).to(torch.float)
	y_train = torch.from_numpy(y_train).to(torch.float)

	x_test = torch.from_numpy(x_test).to(torch.float)
	y_test = torch.from_numpy(y_test).to(torch.float)

	x_valid = torch.from_numpy(x_valid).to(torch.float)
	y_valid = torch.from_numpy(y_valid).to(torch.float)


	# scaled_dataset = pd.DataFrame(scaled_dataset)
	# scaled_dataset.columns = dataset.columns
	print("Shape of train and test datasets of window size of :{}".format(WINDOW_SIZE))
	print("Train x size: {}".format(x_train.shape))
	print("Train y size: {}".format(y_train.shape))	
	print("Valid x size: {}".format(x_valid.shape))
	print("Valid y size: {}".format(y_valid.shape))
	print("Test x size: {}".format(x_test.shape))
	print("Test y size: {}".format(y_test.shape))
	

	train_loader = torch.utils.data.DataLoader(list(zip(x_train,y_train)),batch_size = 8,shuffle = True)
	validation_loader = torch.utils.data.DataLoader(list(zip(x_train,y_train)),batch_size = 8,shuffle = True)


	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# device="cpu"
	learning_rate = 1e-4
	training_epochs = 50
	batch_size = 16
	input_size = WINDOW_SIZE*9
	hidden_size = 10 

	'''optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)'''
	optimizer ="sdg"
	criterion_loss = nn.MSELoss()
	# critetiron_loss = nn.HuberLoss()

	#multiplicative lr
	'''lmbda = lambda training_epochs: 0.65 ** training_epochs'''
	'''scheduler_multiplicative = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)'''
	#cosine anneling
	'''scheduler_annelling = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, training_epochs)'''
	#Exponential
	'''
		torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
	'''
	scheduler_annelling="cosine_anneling"

	model_ann = ANN(	
        input_size = input_size,
        hidden_size = hidden_size,
        optimizer = optimizer,
        criterion = criterion_loss,
        epoch = training_epochs,
        learning_rate = learning_rate,
        device = device)
	model_ann.train_model(train_loader=train_loader,validation_loader=validation_loader)

	max_count = 0
	for entry in os.listdir("models_weight_ann"):
		num = int(entry.split(".")[0].split("_")[1])
		if num > max_count:
			max_count = num

	#save model	
	torch.save(model_ann.state_dict(),"models_weight_ann/ann_{}.pt".format(max_count+1))

	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)

	ax1.title.set_text('Total error')
	ax1.plot(np.arange(len(model_ann.loss)),model_ann.loss)
	ax2.title.set_text('Validation Error')
	ax2.plot(np.arange(len(model_ann.loss_valid)),model_ann.loss_valid)
	plt.show()

	asd = "asd"

	#load model again
	# loadel_model = LSTM_GBRBM(
    #     input_size=input_size,
    #     visible_size=visible_size,
    #     hidden_size=hidden_size,
    #     optimizer = optimizer,
    #     criterion = criterion_loss,
    #     scheduler=scheduler_annelling,
    #     epoch = training_epochs,
    #     clipping = clipping,
    #     k = k,
    #     learning_rate=learning_rate,
    #     cd_step=cd_step,
    #     device=device)

	# loadel_model.load_state_dict(torch.load("models_weight/lstm-gbrbm_{}.pt".format(max_count+1)))
	# loadel_model.eval()

