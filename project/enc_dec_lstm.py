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



class LSTM_encoder(nn.Module):
	def __init__(self,input_size=16,hidden_size=500,n_layers=1,device="cpu"):
		super(LSTM_encoder, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.device = device
		self.n_layers = n_layers
		self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.n_layers, dropout=0.3,batch_first=True).to(self.device)
		self.drop = nn.Dropout(0.2)

	def forward(self,x):
		h_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=torch.float32, requires_grad=True).to(self.device)
		c_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=torch.float32, requires_grad=True).to(self.device)
		out,self.hidden = self.lstm(x, (h_t, c_t))
		# out = self.drop(out)

		return out[:,-1,:], self.hidden 
		# return out,h_n,h_n
	
class LSTM_decoder(nn.Module):
	def __init__(self,input_size=10,hidden_size=500,n_layers=1,device="cpu"):
		super(LSTM_decoder, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.device = device
		self.n_layers = n_layers
		self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.n_layers,dropout=0.3, batch_first=True).to(device)
		self.drop = nn.Dropout(0.2)
		self.linear = nn.Linear(hidden_size, 1).to(device) 

	def forward(self, x_input, encoder_hidden_states):
		'''        
		: param x_input:                    should be 2D (batch_size, input_size)
		: param encoder_hidden_states:      hidden states
		: return output, hidden:            output gives all the hidden states in the sequence;
		:                                   hidden gives the hidden state and cell state for the last
		:                                   element in the sequence 

		'''

		lstm_out, self.hidden = self.lstm(x_input.unsqueeze(1), encoder_hidden_states)
		# lstm_out = self.drop(lstm_out)
		output = self.linear(lstm_out.squeeze(1))     

		return output, self.hidden

class ENC_DEC_LSTM(nn.Module):
	def __init__(self,input_size,hidden_size,optimizer,criterion,epoch,learning_rate,device="cpu"):
		super(ENC_DEC_LSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.criterion = criterion
		self.epoch = epoch
		self.learning_rate = learning_rate
		self.device=device
		self.use_scheduler = True

		# Setting lstm layer and Gaussian Binary Restricted Boltzmann Machine
		self.encoder_layer = LSTM_encoder(input_size=self.input_size,hidden_size=self.hidden_size,device=self.device)
		self.decoder_layer = LSTM_decoder(input_size=input_size,hidden_size=self.hidden_size,device=self.device)
		

		#setting optimizer and learning_rate scheduler
		self.optimizer_lstm = self.get_optimizer(optimizer,"lstm")

		#Setting informazion variables
		self.loss = []
		self.loss_valid = []

	def eval(self):
		self.encoder_layer.eval()
		self.decoder_layer.eval()
		return

	def get_optimizer(self,optimizer_name,type):
		if optimizer_name=="adam":
			return torch.optim.Adam(self.parameters(), lr=self.learning_rate,weight_decay=0.5)
		else:
			return torch.optim.SGD(self.parameters(), lr=self.learning_rate,weight_decay=0.5)

	def train_model(self,train_loader,validation_loader=None):
		for epoch in tqdm(range(self.epoch)):
			print("Current epoch :{}".format(epoch),end="\r")
			loss,loss_valid = self.train_current_epoch(train_loader,validation_loader)
			# return self.train_current_epoch(train_loader)
			self.loss.append(loss)
			self.loss_valid.append(loss_valid)
			print("Current epoch :{} , current error: {}".format(epoch,loss),end="\r")
			print("")

	def predict(self,data):
		encoder_output, encoder_hidden = self.encoder_layer(data)
		outputs = torch.zeros(data.shape[0], 1).to(self.device)

		decoder_input = data[:,-1,:]
		decoder_hidden = encoder_hidden

		for i in range(1):
			decoder_output, decoder_hidden = self.decoder_layer(decoder_input, decoder_hidden)
			# outputs[i] = decoder_output
			outputs[:,i] = decoder_output[:,0]
			decoder_input = decoder_output
		return outputs.detach().numpy()

	def forward(self,data):
		encoder_output, encoder_hidden = self.encoder_layer(data)
		outputs = torch.zeros(data.shape[0], 1).to(self.device)

		decoder_input = data[:,-1,:]
		decoder_hidden = encoder_hidden

		for i in range(1):
			decoder_output, decoder_hidden = self.decoder_layer(decoder_input, decoder_hidden)
			# outputs[i] = decoder_output
			outputs[:,i] = decoder_output[:,0]
			decoder_input = decoder_output

		return outputs


	def train_current_epoch(self,train_loader,validation_loader=None):
		self.train()

		for ii, (data,target)  in enumerate(train_loader):
			self.optimizer_lstm.zero_grad()

			#--------------------------
			data = data.to(self.device)
			target = target.to(self.device)


			# pred = self.forward(data)
			# linear_loss = self.criterion(pred,target)
			outputs = self.forward(data)

			linear_loss = self.criterion(outputs,target)
			linear_loss.backward()



			# pred = self.gbrbm.compute_linear_layer(data_lstm)
			# linear_loss = self.criterion(pred,target)


			# make_dot(pred.mean(), params=dict(self.named_parameters()))

			# grads_lstm = grad(linear_loss,data_lstm)


			# linear_loss.backward()

			self.optimizer_lstm.step()

		validation_error = []
		if validation_loader != None:
			self.eval()
			with torch.no_grad():
				for ii, (data,target)  in enumerate(validation_loader):
					data = data.to(self.device)
					target = target.to(self.device)

					encoder_output, encoder_hidden = self.encoder_layer(data)
					outputs = torch.zeros(target.shape[0], 1).to(self.device)

					decoder_input = data[:,-1,:]
					decoder_hidden = encoder_hidden

					for i in range(1):
						decoder_output, decoder_hidden = self.decoder_layer(decoder_input, decoder_hidden)
						# outputs[i] = decoder_output
						outputs[:,i] = decoder_output[:,0]
						decoder_input = decoder_output

					linear_loss = self.criterion(outputs,target)
					validation_error.append(linear_loss.item())
					

		validation_error = np.array(validation_error).mean()
		return [linear_loss,validation_error]



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
	scaled_dataset = scaler.fit_transform(dataset)
	x_dset,y_dset = create_window_dataset(scaled_dataset,WINDOW_SIZE,y_size=1)

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
	

	train_loader = torch.utils.data.DataLoader(list(zip(x_train,y_train)),batch_size = 16,shuffle = True)
	validation_loader = torch.utils.data.DataLoader(list(zip(x_train,y_train)),batch_size = 16,shuffle = True)


	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# device="cpu"
	learning_rate = 1e-3
	training_epochs = 50
	batch_size = 8
	input_size = 16
	hidden_size = 50 

	'''optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)'''
	optimizer ="adam"
	criterion_loss = nn.MSELoss()

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

	model_lstm_gbrbm = ENC_DEC_LSTM(	
        input_size = input_size,
        hidden_size = hidden_size,
        optimizer = optimizer,
        criterion = criterion_loss,
        epoch = training_epochs,
        learning_rate = learning_rate,
        device = device)
	model_lstm_gbrbm.train_model(train_loader=train_loader,validation_loader=validation_loader)

	max_count = 0
	for entry in os.listdir("models_weight_enc_dec_lstm"):
		num = int(entry.split(".")[0].split("_")[1])
		if num > max_count:
			max_count = num

	#save model	
	torch.save(model_lstm_gbrbm.state_dict(),"models_weight_enc_dec_lstm/enc-dec-lstm_{}.pt".format(max_count+1))

	fig = plt.figure()
	ax1 = fig.add_subplot(311)
	ax2 = fig.add_subplot(312)
	ax3 = fig.add_subplot(313)

	ax1.title.set_text('Total error')
	ax1.plot(np.arange(len(model_lstm_gbrbm.loss)),[e.detach().cpu().item() for e in model_lstm_gbrbm.loss])
	ax3.title.set_text('Validation Error')
	ax3.plot(np.arange(len(model_lstm_gbrbm.loss)),model_lstm_gbrbm.loss_valid)
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
