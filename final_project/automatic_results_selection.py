import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from utils import create_window_dataset,split_train_test,split_train_test_valid,calculate_confusion_matrix
import torch
from torch import nn
from lstm import LSTM
from gru import GRU
from lstm_gbrbm import LSTM_GBRBM
from gru_gbrbm import GRU_GBRBM
from conv_lstm_gbrbm import CONV_LSTM_GBRBM
from conv_gru_gbrbm import CONV_GRU_GBRBM
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,mean_absolute_percentage_error
import numpy as np


# MODEL ="lstm_gbrbm"
# MODEL ="gru_gbrbm"
# MODEL ="lstm"
# MODEL ="gru"rt
# MODELS = ["lstm","gru","lstm_gbrbm","gru_gbrbm"]
MODELS = ["lstm_gbrbm","gru_gbrbm"]
days = [5,20]
# days = [45]
# days = [20]
hidden_sizes = [50,200]
# hidden_sizes = [400]s
batch_sizes = [8,16] 
# batch_sizes = [32]
# drop =[0.1,0.2]
drop =[0.1,0.2]
divs = ["per","div"]
# type_dataset = ["./datasets/stock_prices/nikkei_225/^N225_type1.csv","./datasets/stock_prices/nikkei_225/^N225_type2.csv"]
# type_dataset = ["./datasets/stock_prices/nikkei_225/^BVSP_type1.csv","./datasets/stock_prices/nikkei_225/^BVSP_type2.csv"]
# type_dataset = ["./datasets/stock_prices/nikkei_225/^BVSP_type2.csv"]
# type_dataset = ["./datasets/stock_prices/nikkei_225/^DJI_type1.csv","./datasets/stock_prices/nikkei_225/^DJI_type2.csv"]
# type_dataset = ["./datasets/stock_prices/nikkei_225/FTSEMIB_type1.csv","./datasets/stock_prices/nikkei_225/FTSEMIB_type2.csv"]
# type_dataset = ["./datasets/stock_prices/nikkei_225/SS_type1.csv","./datasets/stock_prices/nikkei_225/SS_type2.csv"]
type_dataset = ["./datasets/stock_prices/nikkei_225/^GDAXI_type1.csv","./datasets/stock_prices/nikkei_225/^GDAXI_type2.csv"]
# type_dataset = ["./datasets/stock_prices/nikkei_225/^GSPC_type1.csv","./datasets/stock_prices/nikkei_225/^GSPC_type2.csv"]
# type_dataset = ["./datasets/stock_prices/nikkei_225/FTSEMIB_type2.csv"]
dset_name = "^GDAXI"







for MODEL in MODELS:
    for DSET_NAME in type_dataset:
        for WINDOW_SIZE in days:
            for HIDDEN_SIZE in hidden_sizes:
                for BATCH_SIZE in batch_sizes:
                    for DROP in drop:
                        for DIV in divs:

                            dataset = pd.read_csv(DSET_NAME)
                            dataset.dropna(inplace=True)
                            dataset.index = dataset["Date"]
                            dataset = dataset.iloc[:,1:]

                            scaler = StandardScaler()
                            scaled_dataset = scaler.fit_transform(dataset)

                            split_size = 0.8
                            valid_size = 0.1

                            x_dset,y_dset = create_window_dataset(scaled_dataset,WINDOW_SIZE,y_size=1)
                            x_train,y_train,x_valid,y_valid,x_test,y_test = split_train_test_valid(x_dset,y_dset,split_size,valid_size)

                            x_train = torch.from_numpy(x_train).to(torch.float)
                            y_train = torch.from_numpy(y_train).to(torch.float)

                            x_test = torch.from_numpy(x_test).to(torch.float)
                            y_test = torch.from_numpy(y_test).to(torch.float)

                            x_valid = torch.from_numpy(x_valid).to(torch.float)
                            y_valid = torch.from_numpy(y_valid).to(torch.float)

                            print("Shape of train and test datasets of window size of :{}".format(WINDOW_SIZE))
                            print("Train x size: {}".format(x_train.shape))
                            print("Train y size: {}".format(y_train.shape))	
                            print("Valid x size: {}".format(x_valid.shape))
                            print("Valid y size: {}".format(y_valid.shape))
                            print("Test x size: {}".format(x_test.shape))
                            print("Test y size: {}".format(y_test.shape))


                            train_loader = torch.utils.data.DataLoader(list(zip(x_train,y_train)),batch_size = BATCH_SIZE,shuffle = True)
                            validation_loader = torch.utils.data.DataLoader(list(zip(x_valid,y_valid)),batch_size = BATCH_SIZE,shuffle = True)
                            test_loader = torch.utils.data.DataLoader(list(zip(x_test,y_test)),batch_size = 1,shuffle = False)
                            
                            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                            optimizer ="adam"
                            criterion_loss = nn.MSELoss()

                            actual_model = ""

                            HIDDEN_GBRBM = int(HIDDEN_SIZE*2) if DIV == "per" else int(HIDDEN_SIZE/2)

                            if MODEL == "gru" and DIV =="div":
                                break
                            if MODEL == "lstm" and DIV == "div":
                                break

                            if MODEL == "gru" and DIV == "per":
                                clipping = 10.0
                                learning_rate_lstm = 1e-4
                                learning_rate_gbrbm = 1e-3
                                training_epochs = 100
                                cd_step = 5
                                batch_size = BATCH_SIZE
                                k = 3
                                input_size = 16 if "type1" in DSET_NAME else 9
                                visible_size = HIDDEN_SIZE
                                hidden_size = HIDDEN_SIZE 

                                scheduler_annelling="cosine_anneling"


                                model_gru = GRU(	
                                    input_size = input_size,
                                    visible_size = visible_size,
                                    hidden_size = hidden_size,
                                    optimizer = optimizer,
                                    criterion = criterion_loss,
                                    scheduler = scheduler_annelling,
                                    epoch = training_epochs,
                                    clipping = clipping,
                                    k = k,
                                    learning_rate_lstm = learning_rate_lstm,
                                    learning_rate_gbrbm = learning_rate_gbrbm,
                                    cd_step = cd_step,
                                    drop=DROP,
                                    device = device)
                                model_gru.train(train_loader=train_loader,validation_loader=validation_loader)


                                torch.save(model_gru.state_dict(),"./final_project/gru/{}/lstm_W{}_H{}_V{}_B{}_D{}.pt"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_SIZE,BATCH_SIZE,str(DROP).split(".")[1]))
                                
                                fig = plt.figure()
                                plt.title("Train lstm_W{}_H{}_V_{}_B{}_D{}"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_SIZE,BATCH_SIZE,str(DROP).split(".")[1]))
                                ax1 = fig.add_subplot(211)
                                ax2 = fig.add_subplot(212)

                                ax1.title.set_text('Total error')
                                ax1.plot(np.arange(len(model_gru.loss)),model_gru.loss)
                                ax2.title.set_text('Validation Error')
                                ax2.plot(np.arange(len(model_gru.loss)),model_gru.loss_valid)
                                plt.savefig("./final_project/gru/{}/train_error_W{}_H{}_V{}_B{}_D{}.png"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_SIZE,BATCH_SIZE,str(DROP).split(".")[1]))
                                actual_model = model_gru

                            if MODEL == "lstm" and DIV == "per":
                                clipping = 10.0
                                learning_rate_lstm = 1e-4
                                learning_rate_gbrbm = 1e-3
                                training_epochs = 100
                                cd_step = 5
                                batch_size = BATCH_SIZE
                                k = 3
                                input_size = 16 if "type1" in DSET_NAME else 9
                                visible_size = HIDDEN_SIZE
                                hidden_size = HIDDEN_SIZE 

                                scheduler_annelling="cosine_anneling"


                                model_lstm = LSTM(	
                                    input_size = input_size,
                                    visible_size = visible_size,
                                    hidden_size = hidden_size,
                                    optimizer = optimizer,
                                    criterion = criterion_loss,
                                    scheduler = scheduler_annelling,
                                    epoch = training_epochs,
                                    clipping = clipping,
                                    k = k,
                                    learning_rate_lstm = learning_rate_lstm,
                                    learning_rate_gbrbm = learning_rate_gbrbm,
                                    cd_step = cd_step,
                                    drop=DROP,
                                    device = device)
                                model_lstm.train(train_loader=train_loader,validation_loader=validation_loader)


                                torch.save(model_lstm.state_dict(),"./final_project/lstm/{}/lstm_W{}_H{}_V{}_B{}_D{}.pt"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_SIZE,BATCH_SIZE,str(DROP).split(".")[1]))
                                
                                fig = plt.figure()
                                plt.title("Train lstm_W{}_H{}_V{}_B{}_D{}"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_SIZE,BATCH_SIZE,str(DROP).split(".")[1]))
                                ax1 = fig.add_subplot(211)
                                ax2 = fig.add_subplot(212)

                                ax1.title.set_text('Total error')
                                ax1.plot(np.arange(len(model_lstm.loss)),model_lstm.loss)
                                ax2.title.set_text('Validation Error')
                                ax2.plot(np.arange(len(model_lstm.loss)),model_lstm.loss_valid)
                                plt.savefig("./final_project/lstm/{}/train_error_W{}_H{}_V{}_B{}_D{}.png"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_SIZE,BATCH_SIZE,str(DROP).split(".")[1]))
                                actual_model = model_lstm

                            if MODEL == "lstm_gbrbm":

                                HIDDEN_GRU = int(HIDDEN_SIZE*2) if DIV == "per" else int(HIDDEN_SIZE/2)

                                clipping = 10.0
                                learning_rate_lstm = 1e-4
                                learning_rate_gbrbm = 1e-3
                                training_epochs = 100
                                cd_step = 5
                                batch_size = BATCH_SIZE
                                k = 3
                                input_size = 16 if "type1" in DSET_NAME else 9
                                visible_size = HIDDEN_SIZE
                                hidden_size = HIDDEN_GRU

                                scheduler_annelling="cosine_anneling"

                                model_lstm_gbrbm = LSTM_GBRBM(	
                                    input_size = input_size,
                                    visible_size = visible_size,
                                    hidden_size = hidden_size,
                                    optimizer = optimizer,
                                    criterion = criterion_loss,
                                    scheduler = scheduler_annelling,
                                    epoch = training_epochs,
                                    clipping = clipping,
                                    k = k,
                                    learning_rate_lstm = learning_rate_lstm,
                                    learning_rate_gbrbm = learning_rate_gbrbm,
                                    cd_step = cd_step,
                                    drop=DROP,
                                    device = device)
                                model_lstm_gbrbm.train(train_loader=train_loader,validation_loader=validation_loader)

                                torch.save(model_lstm_gbrbm.state_dict(),"./final_project/lstm_gbrbm/{}/lstm_gbrbm_W{}_H{}_V{}_B{}_D{}.pt"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_GRU,BATCH_SIZE,str(DROP).split(".")[1]))
                                
                                fig = plt.figure()
                                plt.title("Train lstm_gbrbm_W{}_H{}_V{}_B{}_D{}"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_GRU,BATCH_SIZE,str(DROP).split(".")[1]))
                                ax1 = fig.add_subplot(311)
                                ax2 = fig.add_subplot(312)
                                ax3 = fig.add_subplot(313)

                                ax1.title.set_text('Total error')
                                ax1.plot(np.arange(len(model_lstm_gbrbm.loss)),model_lstm_gbrbm.loss)
                                ax2.title.set_text('GBRBM error')
                                ax2.plot(np.arange(len(model_lstm_gbrbm.loss)),model_lstm_gbrbm.loss_gbrbm)
                                ax3.title.set_text('Validation Error')
                                ax3.plot(np.arange(len(model_lstm_gbrbm.loss)),model_lstm_gbrbm.loss_valid)
                                plt.savefig("./final_project/lstm_gbrbm/{}/train_error_W{}_H{}_V{}_B{}_D{}.png"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_GRU,BATCH_SIZE,str(DROP).split(".")[1]))
                                actual_model = model_lstm_gbrbm
                            
                            if MODEL == "gru_gbrbm":
                                HIDDEN_GBRBM = int(HIDDEN_SIZE*2) if DIV == "per" else int(HIDDEN_SIZE/2)

                                clipping = 10.0
                                learning_rate_lstm = 1e-4
                                learning_rate_gbrbm = 1e-3
                                training_epochs = 100
                                cd_step = 5
                                batch_size = BATCH_SIZE
                                k = 3
                                input_size = 16 if "type1" in DSET_NAME else 9
                                visible_size = HIDDEN_SIZE
                                hidden_size = HIDDEN_GBRBM

                                scheduler_annelling="cosine_anneling"


                                model_gru_gbrbm = GRU_GBRBM(	
                                    input_size = input_size,
                                    visible_size = visible_size,
                                    hidden_size = hidden_size,
                                    optimizer = optimizer,
                                    criterion = criterion_loss,
                                    scheduler = scheduler_annelling,
                                    epoch = training_epochs,
                                    clipping = clipping,
                                    k = k,
                                    learning_rate_lstm = learning_rate_lstm,
                                    learning_rate_gbrbm = learning_rate_gbrbm,
                                    cd_step = cd_step,
                                    drop=DROP,
                                    device = device)
                                
                                model_gru_gbrbm.train(train_loader=train_loader,validation_loader=validation_loader)


                                torch.save(model_gru_gbrbm.state_dict(),"./final_project/gru_gbrbm/{}/gru_gbrbm_W{}_H{}_V{}_B{}_D{}.pt"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_GBRBM,BATCH_SIZE,str(DROP).split(".")[1]))
                                
                                fig = plt.figure()
                                plt.title("Train gru_gbrbm_W{}_H{}_V{}_B{}_D{}"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_GBRBM,BATCH_SIZE,str(DROP).split(".")[1]))
                                ax1 = fig.add_subplot(311)
                                ax2 = fig.add_subplot(312)
                                ax3 = fig.add_subplot(313)

                                ax1.title.set_text('Total error')
                                ax1.plot(np.arange(len(model_gru_gbrbm.loss)),model_gru_gbrbm.loss)
                                ax2.title.set_text('GBRBM error')
                                ax2.plot(np.arange(len(model_gru_gbrbm.loss)),model_gru_gbrbm.loss_gbrbm)
                                ax3.title.set_text('Validation Error')
                                ax3.plot(np.arange(len(model_gru_gbrbm.loss)),model_gru_gbrbm.loss_valid)
                                plt.savefig("./final_project/gru_gbrbm/{}/train_error_W{}_H{}_V{}_B{}_D{}.png"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_GBRBM,BATCH_SIZE,str(DROP).split(".")[1]))
                                actual_model = model_gru_gbrbm
                                
                            if MODEL == "conv_lstm_gbrbm":
                                clipping = 10.0
                                learning_rate_lstm = 1e-4
                                learning_rate_gbrbm = 1e-3
                                training_epochs = 100
                                cd_step = 5
                                batch_size = BATCH_SIZE
                                k = 3
                                input_size = 16 if "type1" in DSET_NAME else 9
                                visible_size = HIDDEN_SIZE
                                hidden_size = int(HIDDEN_SIZE/2)

                                scheduler_annelling="cosine_anneling"

                                conv_lstm_gbrbm = CONV_LSTM_GBRBM(	
                                    input_size = input_size,
                                    visible_size = visible_size,
                                    hidden_size = hidden_size,
                                    optimizer = optimizer,
                                    criterion = criterion_loss,
                                    scheduler = scheduler_annelling,
                                    epoch = training_epochs,
                                    clipping = clipping,
                                    k = k,
                                    learning_rate_lstm = learning_rate_lstm,
                                    learning_rate_gbrbm = learning_rate_gbrbm,
                                    cd_step = cd_step,
                                    device = device,
                                    win_size = WINDOW_SIZE)
                                conv_lstm_gbrbm.train(train_loader=train_loader,validation_loader=validation_loader)

                                torch.save(conv_lstm_gbrbm.state_dict(),"./final_project/conv_lstm_gbrbm/{}/conv_lstm_gbrbm_W{}_H{}_B{}_D{}.pt"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,BATCH_SIZE,str(DROP).split(".")[1]))
                                
                                fig = plt.figure()
                                plt.title("Train conv_lstm_gbrbm_W{}_H{}_B{}_D{}"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,BATCH_SIZE,str(DROP).split(".")[1]))
                                ax1 = fig.add_subplot(311)
                                ax2 = fig.add_subplot(312)
                                ax3 = fig.add_subplot(313)

                                ax1.title.set_text('Total error')
                                ax1.plot(np.arange(len(conv_lstm_gbrbm.loss)),conv_lstm_gbrbm.loss)
                                ax2.title.set_text('GBRBM error')
                                ax2.plot(np.arange(len(conv_lstm_gbrbm.loss)),conv_lstm_gbrbm.loss_gbrbm)
                                ax3.title.set_text('Validation Error')
                                ax3.plot(np.arange(len(conv_lstm_gbrbm.loss)),conv_lstm_gbrbm.loss_valid)
                                plt.savefig("./final_project/conv_lstm_gbrbm/{}/train_error_W{}_H{}_B{}_D{}.png"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,BATCH_SIZE,str(DROP).split(".")[1]))                       
                                actual_model = conv_lstm_gbrbm

                            if MODEL == "conv_gru_gbrbm":
                                clipping = 10.0
                                learning_rate_lstm = 1e-4
                                learning_rate_gbrbm = 1e-3
                                training_epochs = 100
                                cd_step = 5
                                batch_size = BATCH_SIZE
                                k = 3
                                input_size = 16 if "type1" in DSET_NAME else 9
                                visible_size = HIDDEN_SIZE
                                hidden_size = int(HIDDEN_SIZE/2)

                                scheduler_annelling="cosine_anneling"

                                conv_gru_gbrbm = CONV_GRU_GBRBM(	
                                    input_size = input_size,
                                    visible_size = visible_size,
                                    hidden_size = hidden_size,
                                    optimizer = optimizer,
                                    criterion = criterion_loss,
                                    scheduler = scheduler_annelling,
                                    epoch = training_epochs,
                                    clipping = clipping,
                                    k = k,
                                    learning_rate_lstm = learning_rate_lstm,
                                    learning_rate_gbrbm = learning_rate_gbrbm,
                                    cd_step = cd_step,
                                    device = device,
                                    win_size = WINDOW_SIZE)
                                conv_gru_gbrbm.train(train_loader=train_loader,validation_loader=validation_loader)

                                torch.save(conv_gru_gbrbm.state_dict(),"./final_project/conv_gru_gbrbm/{}/conv_gru_gbrbm_W{}_H{}_B{}_D{}.pt"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,BATCH_SIZE,str(DROP).split(".")[1]))
                                
                                fig = plt.figure()
                                plt.title("Train conv_gru_gbrbm_W{}_H{}_B{}_D{}"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,BATCH_SIZE,str(DROP).split(".")[1]))
                                ax1 = fig.add_subplot(311)
                                ax2 = fig.add_subplot(312)
                                ax3 = fig.add_subplot(313)

                                ax1.title.set_text('Total error')
                                ax1.plot(np.arange(len(conv_gru_gbrbm.loss)),conv_gru_gbrbm.loss)
                                ax2.title.set_text('GBRBM error')
                                ax2.plot(np.arange(len(conv_gru_gbrbm.loss)),conv_gru_gbrbm.loss_gbrbm)
                                ax3.title.set_text('Validation Error')
                                ax3.plot(np.arange(len(conv_gru_gbrbm.loss)),conv_gru_gbrbm.loss_valid)
                                plt.savefig("./final_project/conv_gru_gbrbm/{}/train_error_W{}_H{}_B{}_D{}.png"
                                        .format(DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,BATCH_SIZE,str(DROP).split(".")[1]))                       
                                actual_model = conv_gru_gbrbm


                            actual_model.eval()

                            x_axis = np.arange(0,x_test.shape[0])
                            y_pred_train = []
                            y_actual_train = []
                            y_pred_valid = []
                            y_actual_valid = []
                            y_pred_test = []
                            y_actual_test = []

                            loss_train = []
                            loss_valid = []
                            loss_test = []

                            train_loader = torch.utils.data.DataLoader(list(zip(x_train,y_train)),batch_size = 1,shuffle = False)
                            validation_loader = torch.utils.data.DataLoader(list(zip(x_valid,y_valid)),batch_size = 1,shuffle = False)
                            

                            #info about train
                            for ii, (data,target)  in enumerate(train_loader):
                                data = data.to(actual_model.device)
                                target = target.to(actual_model.device) 

                                pred = actual_model.forward(data)

                                linear_loss = actual_model.criterion(pred,target)
                                loss_train.append(linear_loss.item())

                                y_pred_train.append(pred[0].detach().to("cpu").numpy()[0])
                                y_actual_train.append(target[0].detach().to("cpu").numpy()[0])
                            
                            #info about valid
                            for ii, (data,target)  in enumerate(validation_loader):
                                data = data.to(actual_model.device)
                                target = target.to(actual_model.device) 

                                pred = actual_model.forward(data)

                                linear_loss = actual_model.criterion(pred,target)
                                loss_valid.append(linear_loss.item())

                                y_pred_valid.append(pred[0].detach().to("cpu").numpy()[0])
                                y_actual_valid.append(target[0].detach().to("cpu").numpy()[0])

                            #info about test
                            for ii, (data,target)  in enumerate(test_loader):
                                data = data.to(actual_model.device)
                                target = target.to(actual_model.device) 

                                pred = actual_model.forward(data)

                                linear_loss = actual_model.criterion(pred,target)
                                loss_test.append(linear_loss.item())

                                y_pred_test.append(pred[0].detach().to("cpu").numpy()[0])
                                y_actual_test.append(target[0].detach().to("cpu").numpy()[0])


                            trend_pred1 = []
                            trend_pred = []
                            trend_actual = []

                            #trend analysis
                            for (idx,el) in enumerate(y_pred_test):
                                if idx > 0:
                                    y_prd1 = 1 if y_pred_test[idx] - y_pred_test[idx-1] > 0 else 0
                                    y_prd = 1 if y_pred_test[idx] - y_actual_test[idx-1] > 0 else 0
                                    y_act = 1 if y_actual_test[idx] - y_actual_test[idx-1] > 0 else 0

                                    trend_pred1.append(y_prd1)
                                    trend_pred.append(y_prd)
                                    trend_actual.append(y_act)

                            trend_pred1 = np.array(trend_pred1)
                            trend_pred = np.array(trend_pred)
                            trend_actual = np.array(trend_actual)

                            tp,tn,fp,fn = calculate_confusion_matrix(trend_actual,trend_pred)
                            precision_pos = tp/(tp+fp)
                            precision_neg = tn/(tn+fn)
                            recall_pos = tp/(tp+fn)
                            recall_ne = tn/(tn+fp)
                            accuracy = (tp+tn)/(tp+fp+tn+fn)
                            f1_score = 2*precision_pos*recall_pos/(precision_pos+recall_pos)

                            y_pred_trans = scaler.inverse_transform(np.concatenate([dataset.iloc[-x_test.shape[0]:,:-1],np.expand_dims(np.array(y_pred_test),axis=1)],axis=1))[:,-1:][:,0]
                            y_actual_trans = scaler.inverse_transform(np.concatenate([dataset.iloc[-x_test.shape[0]:,:-1],np.expand_dims(np.array(y_actual_test),axis=1)],axis=1))[:,-1:][:,0]

                            total_mape = mean_absolute_percentage_error(y_actual_trans,y_pred_trans)
                            total_mae = mean_absolute_error(y_actual_trans,y_pred_trans)
                            total_mse = mean_squared_error(y_actual_trans,y_pred_trans)
                            total_rmse = mean_squared_error(y_actual_trans,y_pred_trans,squared=False)
                            total_r2 = r2_score(y_actual_trans,y_pred_trans)

                            pd_results = pd.DataFrame({"precision_pos":[precision_pos],
                                                    "precision_neg":[precision_neg],
                                                    "recall_pos":[recall_pos],
                                                    "recall_neg":[recall_ne],
                                                    "accuracy":[accuracy],
                                                    "f1_score":[f1_score],
                                                    "accuracy2":[accuracy_score(trend_actual,trend_pred1)],
                                                    "--":["--"],
                                                    "mape":[total_mape],
                                                    "mae":[total_mae],
                                                    "mse":[total_mse],
                                                    "rmse":[total_rmse],
                                                    "r2":[total_r2]})
                            
                            results_path = ""
                            # plot_test_train_path = ""
                            # error_test_train_path = ""

                            results_path = "./final_project/{}/{}/pd_results_W{}_H{}_V{}_B{}_D{}.xlsx"\
                            .format(MODEL,DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_GBRBM,BATCH_SIZE,str(DROP).split(".")[1])
                            # plot_test_train_path = "final_project/{}/{}/plot_train_test{}_H{}_B{}.png"\
                            # .format(MODEL,DSET_NAME,WINDOW_SIZE,HIDDEN_SIZE,BATCH_SIZE)
                            # error_test_train_path = "final_project/{}/{}/train_test_error{}_H{}_B{}.png"\
                            # .format(MODEL,DSET_NAME,WINDOW_SIZE,HIDDEN_SIZE,BATCH_SIZE)

                            pd_results.to_excel(results_path,index=False)

                            #plot error test and error train
                            fig = plt.figure()
                            ax1 = fig.add_subplot(311)
                            ax2 = fig.add_subplot(312)
                            ax3 = fig.add_subplot(313)
                            ax1.title.set_text('Train error')
                            ax1.plot(np.arange(len(loss_train)),loss_train)
                            ax2.title.set_text('Valid error')
                            ax2.plot(np.arange(len(loss_valid)),loss_valid)
                            ax3.title.set_text('Test error')
                            ax3.plot(np.arange(len(loss_test)),loss_test)
                            plt.savefig("./final_project/{}/{}/test_error_W{}_H{}_V{}_B{}_D{}.png"
                                        .format(MODEL,DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_GBRBM,BATCH_SIZE,str(DROP).split(".")[1]))
                            
                        


                            #plot train valid test pred
                            fig = plt.figure()
                            ax1 = fig.add_subplot(311)
                            ax2 = fig.add_subplot(312)
                            ax3 = fig.add_subplot(313)
                            ax1.title.set_text('Train prediction')
                            ax1.plot(np.arange(len(y_pred_train)),y_pred_train)
                            ax1.plot(np.arange(len(y_actual_train)),y_actual_train)
                            ax1.legend(loc="upper right")
                            ax2.title.set_text('Valid prediction')
                            ax2.plot(np.arange(len(y_pred_valid)),y_pred_valid)
                            ax2.plot(np.arange(len(y_actual_valid)),y_actual_valid)
                            ax2.legend(loc="upper right")
                            ax3.title.set_text('Test prediction')
                            ax3.plot(np.arange(len(y_pred_test)),y_pred_test)
                            ax3.plot(np.arange(len(y_actual_test)),y_actual_test)
                            ax3.legend(loc="upper right")
                            plt.savefig("./final_project/{}/{}/plot_test_W{}_H{}_V{}_B{}_D{}.png"
                                        .format(MODEL,DSET_NAME.split(dset_name+"_")[1].split(".")[0],WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_GBRBM,BATCH_SIZE,str(DROP).split(".")[1]))




	
                  




