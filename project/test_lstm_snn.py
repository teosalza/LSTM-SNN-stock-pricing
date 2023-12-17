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
import os
from utils import create_window_dataset,split_train_test
from lstn_snn import LSTM_GBRBM


if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(description="LSTM-SNN test.", epilog="""-------------------""")

    # Required positional arguments
    parser.add_argument("--dataset-name", type=str,
                        help="[string] specify the name.",required=True)
    parser.add_argument("--window-size", type=int,
                        help="[int] specify the window size.",required=True)
    parser.add_argument("--weight-dir", type=str,
                        help="[string] specify weight directory .",required=False)

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    DATASET_NAME = args.dataset_name
    WINDOW_SIZE = args.window_size
    WEIGHT_DIR = args.weight_dir

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

    train_loader_test = torch.utils.data.DataLoader(list(zip(x_test,y_test)),batch_size = 1,shuffle = False)
    model_weight_path = ""
    max_int = 0
    for entry in os.listdir(WEIGHT_DIR):
        number = int(entry.split(".")[0].split("_")[1])
        if number > max_int:
            max_int = number
            model_weight_path = WEIGHT_DIR+"/"+entry




    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    clipping = 10.0
    learning_rate_lstm = 1e-3
    learning_rate_gbrbm = 1e-4
    training_epochs = 15
    cd_step = 10
    batch_size = 1
    k = 3      
    input_size=16
    visible_size = 500
    hidden_size = 250

    '''optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)'''
    optimizer ="sdg"
    criterion_loss = nn.MSELoss()

    #multiplicative lr
    lmbda = lambda training_epochs: 0.65 ** training_epochs
    '''scheduler_multiplicative = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)'''
    #cosine anneling
    '''scheduler_annelling = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, training_epochs)'''
    #Exponential
    '''
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    '''
    scheduler_annelling="exponential_lr"

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
        learning_rate_lstm=learning_rate_lstm,
        learning_rate_gbrbm=learning_rate_gbrbm,
        cd_step=cd_step,
        device=device)

    model_lstm_gbrbm.load_state_dict(torch.load(model_weight_path))
    model_lstm_gbrbm.eval()

    x_axis = np.arange(0,x_test.shape[0])
    y_axis = []
    y_actual=[]

    loss_array = []

    #test section
    for ii, (data,target)  in enumerate(train_loader_test):
        data = data.to(model_lstm_gbrbm.device)
        target = target.to(model_lstm_gbrbm.device) 

        pred = model_lstm_gbrbm.forward(data)
        pred = pred[None,:]

        linear_loss = model_lstm_gbrbm.criterion(pred,target)
        loss_array.append(linear_loss.item())
        # y_axis.append(pred.mean().item())
        # y_actual.append(target.mean().item())
        y_axis.append(pred[0][0][0].item())
        y_actual.append(target[0][0].item())

    plt.plot(x_axis,y_axis)
    plt.plot(x_axis,y_actual)
    plt.show()

    plt.plot(np.arange(len(loss_array)),loss_array)
    plt.show()
