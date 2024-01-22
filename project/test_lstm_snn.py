import torch
from torch import nn
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,mean_absolute_percentage_error
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
from scipy.ndimage import shift
from sklearn.metrics import accuracy_score


def calculate_confusion_matrix(pred,actual):
    negative = 0
    positive = 1
    pred = np.concatenate([pred[0],np.array(pred[1:]).flatten()])
    actual = np.concatenate([actual[0],np.array(actual[1:]).flatten()])
    # pred = np.array(pred[1:]).flatten()
    # pred = np.array(actual[1:]).flatten()
    # pred = np.array(pred)
    # actual = np.array(actual)
    tp = np.sum(np.logical_and(pred == positive, actual == positive))
    tn = np.sum(np.logical_and(pred == negative, actual == negative))
    fp = np.sum(np.logical_and(pred == positive, actual == negative))
    fn = np.sum(np.logical_and(pred == negative, actual == positive))
    return tp,tn,fp,fn

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
    # scaled_dataset = scaler.fit_transform(dataset)
    normal = MinMaxScaler(feature_range=(-1, 1))
    scaled_dataset = normal.fit_transform(dataset)

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
    train_loader = torch.utils.data.DataLoader(list(zip(x_train,y_train)),batch_size = 1,shuffle = False)
    model_weight_path = ""
    max_int = 0

    for entry in os.listdir(WEIGHT_DIR):
        number = int(entry.split(".")[0].split("_")[1])
        if number > max_int:
            max_int = number
            model_weight_path = WEIGHT_DIR+"/"+entry

    # model_weight_path = WEIGHT_DIR + "/"+"lstm-gbrbm_64.pt"
    




    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    clipping = 10.0
    learning_rate_lstm = 1e-3
    learning_rate_gbrbm = 1e-4
    training_epochs = 15
    cd_step = 2
    batch_size = 1
    k = 3      
    input_size=9
    visible_size = 50
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
    y_pred = []
    y_actual=[]

   
    # for ii, (data,target)  in enumerate(train_loader_test):
    #     data = data.to(model_lstm_gbrbm.device)
    #     target = target.to(model_lstm_gbrbm.device) 

    #     pred = model_lstm_gbrbm.forward(data)

    #     linear_loss = model_lstm_gbrbm.criterion(pred,target)
    #     loss_array.append(linear_loss.item())
    #     # y_axis.append(pred.mean().item())
    #     # y_actual.append(target.mean().item())
    #     if one_day_ahead:
    #         y_pred.append(pred[0].detach().to("cpu").numpy()[0])
    #         y_actual.append(target[0].detach().to("cpu").numpy()[0])
    #     else:
    #         y_pred.append(pred[0].detach().to("cpu").numpy())
    #         y_actual.append(target[0].detach().to("cpu").numpy())

       
    #     pred_trend = []
    #     actual_trend = []
    #     if one_day_ahead:
    #         max_range = 1
    #     else:
    #         max_range = 3
            
        
    #     for i in range(max_range):
    #         if i == 0:
    #             if ii > 0:
    #                 if old_pred[0][0].item()-pred[0][i].item() > 0:
    #                     pred_trend.append(1)
    #                 else:
    #                     pred_trend.append(0)

    #                 if old_actual[0][0].item()-target[0][i].item() > 0:
    #                     actual_trend.append(1)
    #                 else:
    #                     actual_trend.append(0)
    #         else:
    #             if pred[0][i-1].item()-pred[0][i].item() > 0:
    #                     pred_trend.append(1)
    #             else:
    #                 pred_trend.append(0)
            
    #             if target[0][i-1].item()-target[0][i].item() > 0:
    #                 actual_trend.append(1)
    #             else:
    #                 actual_trend.append(0)
        
    #     trend_pred.append(pred_trend)
    #     trend_actual.append(actual_trend)

    #     old_pred = target
    #     old_actual = target
       


    trend_pred =[]
    trend_actual =[]

    loss_array = []

    old_pred = []
    old_actual = []
    #test section
    one_day_ahead = True
    for ii, (data,target)  in enumerate(train_loader_test):
        

        data = data.to(model_lstm_gbrbm.device)
        target = target.to(model_lstm_gbrbm.device) 

        pred = model_lstm_gbrbm.forward(data)

        linear_loss = model_lstm_gbrbm.criterion(pred,target)
        loss_array.append(linear_loss.item())
        # y_axis.append(pred.mean().item())
        # y_actual.append(target.mean().item())
        if one_day_ahead:
            y_pred.append(pred[0].detach().to("cpu").numpy()[0])
            y_actual.append(target[0].detach().to("cpu").numpy()[0])
        else:
            y_pred.append(pred[0].detach().to("cpu").numpy())
            y_actual.append(target[0].detach().to("cpu").numpy())

       
        pred_trend = []
        actual_trend = []
        if one_day_ahead:
            max_range = 1
        else:
            max_range = 3
            
        
        for i in range(max_range):
            if i == 0:
                if ii > 0:
                    if old_pred[0][0].item()-pred[0][i].item() > 0:
                        pred_trend.append(1)
                    else:
                        pred_trend.append(0)

                    if old_actual[0][0].item()-target[0][i].item() > 0:
                        actual_trend.append(1)
                    else:
                        actual_trend.append(0)
            else:
                if pred[0][i-1].item()-pred[0][i].item() > 0:
                        pred_trend.append(1)
                else:
                    pred_trend.append(0)
            
                if target[0][i-1].item()-target[0][i].item() > 0:
                    actual_trend.append(1)
                else:
                    actual_trend.append(0)
        
        trend_pred.append(pred_trend)
        trend_actual.append(actual_trend)

        old_pred = target
        old_actual = target
       

    # trend_pred = [[1] if el > 0 else [0] for el in (shift(y_pred,-1,cval=np.NaN) - y_pred)]
    # trend_actual = [[1] if el > 0 else [0] for el in (shift(y_actual,-1,cval=np.NaN) - y_actual)]

    trend_actual1 = [0 if el < 0 else 1 for el in (np.array(y_actual) - np.roll(np.array(y_actual),1))] 
    trend_pred1 = [0 if el < 0 else 1 for el in (np.array(y_pred) - np.roll(np.array(y_pred),1))] 

    tp,tn,fp,fn = calculate_confusion_matrix(trend_pred,trend_actual)
    precision_pos = tp/(tp+fp)
    precision_neg = tn/(tn+fn)
    recall_pos = tp/(tp+fn)
    recall_ne = tn/(tn+fp)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    f1_score = 2*precision_pos*recall_pos/(precision_pos+recall_pos)
    print("Precision pos: {}".format(precision_pos))
    print("Precision neg: {}".format(precision_neg))
    print("Recall pos: {}".format(recall_pos))  
    print("Recall neg: {}".format(recall_ne))
    print("Accuracy: {}".format(accuracy))
    print("F1 score: {}".format(f1_score))





    # plt.plot(x_axis[100:300],y_pred[100:300],label="Predicted")   
    # plt.plot(x_axis[100:300],y_actual[100:300],label="Actual")
    plt.plot(x_axis,y_pred,label="Predicted")   
    plt.plot(x_axis,y_actual,label="Actual")
    plt.legend(loc="upper left")
    plt.show()


    y_pred_transformed = scaler.inverse_transform(np.concatenate([dataset.iloc[-x_test.shape[0]:,:-1],np.expand_dims(np.array(y_pred),axis=1)],axis=1))[:,-1:]
    y_actual_transformed = scaler.inverse_transform(np.concatenate([dataset.iloc[-x_test.shape[0]:,:-1],np.expand_dims(np.array(y_actual),axis=1)],axis=1))[:,-1:]

    #for 1 day ahead
    y_pred_transformed = y_pred_transformed[:,0]
    y_actual_transformed = y_actual_transformed[:,0]

    total_mape = mean_absolute_percentage_error(y_actual_transformed,y_pred_transformed)
    total_mae = mean_absolute_error(y_actual_transformed,y_pred_transformed)
    total_mse = mean_squared_error(y_actual_transformed,y_pred_transformed)
    total_rmse = mean_squared_error(y_actual_transformed,y_pred_transformed,squared=False)
    total_r2 = r2_score(y_actual_transformed,y_pred_transformed)
    print("MAPE: {}".format(total_mape*100))
    print("MAE: {}".format(total_mae))
    print("MSE: {}".format(total_mse))
    print("RMSE: {}".format(total_rmse))
    print("R2: {}".format(total_r2))
    plt.plot(np.arange(len(loss_array)),loss_array)
    plt.show()

    plt.plot(x_axis,y_pred_transformed[:,0],label="Predicted")
    plt.plot(x_axis,y_actual_transformed[:,0],label="Actual")
    plt.legend(loc="upper left")
    plt.show()
    a=""
