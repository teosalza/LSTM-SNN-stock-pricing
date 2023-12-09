import torch
from torch import nn
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
import argparse
import tqdm
import matplotlib.pyplot as plt
from torchviz import make_dot, make_dot_from_trace
import torch.nn.functional as F
from torch.autograd import grad
import pandas as pd



class GBRBM(torch.nn.Module):   
	def __init__(self,visible_size,hidden_size,init_var=1e-0,cd_step=1,cd_burning=0,device="cpu"):
		super().__init__()
		self.cd_step = cd_step
		self.cd_burning = cd_burning
		self.visible_size = visible_size
		self.hidden_size = hidden_size
		self.init_var = init_var
		self.device=device
		
		self.linear_layer = nn.Linear(hidden_size,3).to(self.device)
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
		prob_h, h, pos_eng = self.positive_energy(v) 

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
	def positive_energy(self, v):
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
		return self.linear_layer(prob_h[0])
	
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
		return self.linear_layer(prob_h[0])

class LSTM_module(nn.Module):
	def __init__(self,input_size=10,hidden_size=500,device="cpu"):
		super(LSTM_module, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.device = device
		self.lstm1 = nn.LSTM(self.input_size, self.hidden_size,batch_first=True).to(device)

	def forward(self,x):
		h_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=torch.float32, requires_grad=True).to(self.device)
		c_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=torch.float32, requires_grad=True).to(self.device)
		out,(h_n,c_n) = self.lstm1(x, (h_t, c_t))
		
		return out[:,-1,:]
		# return out,h_n,h_n

class LSTM_GBRBM(nn.Module):
	def __init__(self,input_size,visible_size,hidden_size,optimizer,criterion,scheduler,epoch,clipping,k,learning_rate,cd_step,device="cpu"):
		super(LSTM_GBRBM, self).__init__()
		self.input_size = input_size
		self.visible_size = visible_size
		self.hidden_size = hidden_size
		self.criterion = criterion
		self.epoch = epoch
		self.clipping = clipping
		self.k = k
		self.learning_rate = learning_rate
		self.cd_step = cd_step
		self.device=device
		self.dot = ""

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
		self.optimizer = self.get_optimizer(optimizer)
		self.scheduler = self.get_scheduler(scheduler)

		#Setting informazion variables
		self.loss = []
		self.loss_gbrbm = []
		self.lr_list = []

	def get_optimizer(self,optimizer_name):
		if optimizer_name=="adam":
			return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		if optimizer_name=="sdg":
			return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
		return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

	def get_scheduler(self,scheduler_name):
		if scheduler_name == "cosine_anneling":
			return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epoch)
		return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epoch)
	
	def train(self,train_loader):
		for epoch in range(self.epoch):
			print("Current epoch :{}".format(epoch),end="\r")
			loss, loss_gbrbm = self.train_current_epoch(train_loader)
			# return self.train_current_epoch(train_loader)
			self.loss.append(loss)
			self.loss_gbrbm.append(loss_gbrbm)
			self.lr_list.append(self.optimizer.param_groups[0]["lr"])
			self.scheduler.step()
			print("Current epoch :{} , current error: {}, current error gbrmb: {}".format(epoch,loss,loss_gbrbm),end="\r")
			print("")

	def predict(self,data):
		data_lstm = self.lstm_layer(data)
		pred = self.gbrbm.compute_linear_layer(data_lstm)
		return pred.detach().numpy()

	def forward(self,data):
		data_lstm = self.lstm_layer(data)
		pred = self.gbrbm(data_lstm)
		return pred

	def train_current_epoch(self,train_loader):
		self.lstm_layer.train()
		self.gbrbm.train()

		for ii, (data,target)  in enumerate(train_loader):
			self.optimizer.zero_grad()
			
			#--------------------------
			data = data.to(self.device)
			target = target.to(self.device)
			
			pred = self.forward(data)
			linear_loss = self.criterion(pred,target)
			
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

			# linear_loss.backward()
			
			self.optimizer.step()
			if ii == len(train_loader) - 1:
				recon_loss = self.gbrbm.reconstruction(data_lstm).item()

		return [recon_loss,linear_loss]

def create_window_dataset(dataset,wind_size):
    x_wind = []
    y_wind = []
    for i in range(wind_size,dataset.shape[0]):
        curr = dataset[i-wind_size:i]
        x_wind.append(curr[:,:-3])
        y_wind.append(curr[-1,-3:])
    return np.array(x_wind),np.array(y_wind)

def split_train_test(x_dset,y_dset,split_size):
    test_set_size = int(np.round(0.2*x_dset.shape[0]))  
    train_set_size = x_dset.shape[0] - (test_set_size)
    return x_dset[:train_set_size],y_dset[:train_set_size],x_dset[train_set_size:],y_dset[train_set_size:]




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
    x_dset,y_dset = create_window_dataset(scaled_dataset,WINDOW_SIZE)

    split_size = 0.8
    x_train,y_train,x_test,y_test = split_train_test(x_dset,y_dset,split_size)

    x_train = torch.from_numpy(x_train).to(torch.float)
    y_train = torch.from_numpy(y_train).to(torch.float)
    
    x_test = torch.from_numpy(x_test).to(torch.float)
    y_test = torch.from_numpy(y_test).to(torch.float)

    # scaled_dataset = pd.DataFrame(scaled_dataset)
    # scaled_dataset.columns = dataset.columns
    print("Shape of train and test datasets of window size of :{}".format(WINDOW_SIZE))
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape) 
    print(y_test.shape)

    train_loader = torch.utils.data.DataLoader(list(zip(x_train,y_train)),batch_size = 1,shuffle = False)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    clipping = 10.0
    learning_rate = 1e-4
    training_epochs = 10
    cd_step = 5
    batch_size = 1
    k = 3      
    input_size=16
    visible_size = 500
    hidden_size = 25

    '''optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)'''
    optimizer ="sdg"
    criterion_loss = nn.MSELoss()

    #multiplicative lr
    lmbda = lambda training_epochs: 0.65 ** training_epochs
    '''scheduler_multiplicative = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)'''
    #cosine anneling
    '''scheduler_annelling = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, training_epochs)'''
    scheduler_annelling="cosine_anneling"

    model_lstm_gbrbm = LSTM_GBRBM(
        input_size=input_size,
        visible_size=visible_size,
        hidden_size=hidden_size,
        optimizer = optimizer,
        criterion = criterion_loss,
        scheduler=scheduler_annelling,
        epoch = training_epochs,
        clipping = clipping,
        k = k,
        learning_rate=learning_rate,
        cd_step=cd_step,
        device=device)

    model_lstm_gbrbm.train(train_loader=train_loader)
    print(scaled_dataset)
