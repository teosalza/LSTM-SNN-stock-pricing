
import numpy as np 

def create_window_dataset(dataset,wind_size):
    x_wind = []
    y_wind = []
    for i in range(wind_size,dataset.shape[0]):
        curr = dataset[i-wind_size:i]
        x_wind.append(curr[:,:-1])
        y_wind.append(curr[-1,-1:])
    return np.array(x_wind),np.array(y_wind)


def split_train_test(x_dset,y_dset,split_size):
    train_set_size = int(np.round(split_size*x_dset.shape[0]))  
    test_set_size = x_dset.shape[0] - (train_set_size)
    return x_dset[:train_set_size],y_dset[:train_set_size],x_dset[-test_set_size:],y_dset[-test_set_size:]

def split_train_test_valid(x_dset,y_dset,split_size,valid_size):
    train_set_size = int(np.round(split_size*x_dset.shape[0])) 
    valid_set_size = int(np.round(valid_size*x_dset.shape[0]))
    test_set_size = x_dset.shape[0] - (train_set_size)
    train_set_size -= valid_set_size
    return x_dset[:train_set_size],y_dset[:train_set_size],x_dset[train_set_size:train_set_size+valid_set_size],y_dset[train_set_size:train_set_size+valid_set_size],x_dset[-test_set_size:],y_dset[-test_set_size:]

